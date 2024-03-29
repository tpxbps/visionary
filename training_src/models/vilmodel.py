import json 
import logging
import math 
import os 
import sys 
from io import open
from typing import Callable, List, Tuple
import numpy as np
import copy 
import torch 
import h5py 
from torch import nn 
import torch.nn.functional as F 
from torch import Tensor, device, dtype 
from transformers import BertPreTrainedModel 
from warmup_src.model.ops import extend_neg_masks, gen_seq_masks, pad_tensors_wgrad, create_transformer_encoder

from warmup_src.model.vilmodel import CrossmodalEncoder

CROP_SIZE = 5
logger = logging.getLogger(__name__)
BertLayerNorm = torch.nn.LayerNorm 


ROOM2IND = {'balcony' : 23, 'bathroom': 0, 'classroom': 26, 'dining_booth': 27, 'entryway': 4, 'garage': 6, 
            'junk': 29, 'laundryroom': 9, 'living room': 11, 'meetingroom': 12, 'other_room': 24, 'porch': 15,
            'spa': 28, 'toilet': 18, 'utilityroom': 19, 'bar': 25, 'bedroom': 1, 'closet': 2,
            'dining_room': 3, 'familyroom': 5, 'hallway': 7, 'kitchen': 10, 'library': 8, 'lounge': 13,
            'office': 14, 'outdoor': 22,'rec': 16, 'stairs': 17, 'tv': 20, 'workout': 21,
}

IND2ROOM={ 23: 'balcony', 0: 'bathroom', 26: 'classroom', 27: 'dining_booth',4: 'entryway', 6: 'garage',
           29: 'junk', 9: 'laundryroom', 11: 'living room', 12: 'meetingroom',24:'other_room', 15: 'porch',
           28: 'spa', 18: 'toilet', 19 :'utilityroom', 25:'bar', 1:'bedroom', 2: 'closet',
           3: 'dining_room', 5: 'familyroom', 7 :'hallway', 10: 'kitchen', 8:'library', 13: 'lounge',
           14: 'office', 12: 'outdoor', 16: 'rec', 17: 'stairs',  20:'tv', 21: 'workout'
}


def gelu(x):
    """Implementationo of the gelu activation function.
       0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x,3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {'gelu': gelu, 'relu': torch.nn.functional.relu, 'swish': swish}

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type_embeddings
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask, head_mask=None):
        """
        hidden_states: (N, L_{hidden}, D)
        attention_mask: (N, H, L_{hidden}, L_{hidden})

        TODO: notice --> attention_mask must be computed beforehand 
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # recurrent vlnbert use attention scores
        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
    
    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask,
                None if head_mask is None else head_mask[i],
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOutAttention(nn.Module):
    """
    Different BertAttention ->
    hear k,v => context q => hiddenstate -> they could have different dimensions
    mainly used when we do cross model attention 

    """
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores


class BertXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output, attention_scores = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output, attention_scores


class LanguageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_l_layers = config.num_l_layers
        self.update_lang_bert = config.update_lang_bert

        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False

    def forward(self, txt_embeds, txt_masks):
        extended_txt_masks = extend_neg_masks(txt_masks)
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]
        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()
        return txt_embeds

class CLIPLanguageEncoder(nn.Module):
    def __init__(self, config):
        pass 


class GlobalVisualEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_v_layers = config.num_v_layers 
        self.layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_v_layers)]
        )
    
    def forward(self, vis_embeds, vis_masks, graph_sprels=None):
        visn_attention_mask = extend_neg_masks(vis_masks)
        if graph_sprels is not None:
            visn_attention_mask = visn_attention_mask + graph_sprels
        for layer_module in self.layers:
            temp_output = layer_module(vis_embeds, visn_attention_mask)
            vis_embeds = temp_output[0]

        return vis_embeds 

class LayoutGraphAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layout_layers = config.num_layout_layers
        self.layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_layout_layers)]
        )
    
    def forward(self, layout_map_embeds, layout_map_masks, graph_sprels=None):
        map_masks = extend_neg_masks(layout_map_masks)
        
        if graph_sprels is not None:
            map_masks = map_masks + graph_sprels
        for layer_module in self.layers:
            temp_output = layer_module(layout_map_embeds, map_masks)
            layout_map_embeds = temp_output[0]
        return layout_map_embeds


class LayoutGrpahX(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = CrossmodalEncoder(config)

    def forward(self, txt_embeds, txt_masks, gmap_embeds,gmap_masks, graph_sprels=None):
        
        gmap_embeds = self.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks,
            graph_sprels = graph_sprels
        )

        return gmap_embeds


class GraphLXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Lang self-att and FFN layer
        if config.use_lang2visn_attn:
            self.lang_self_att = BertAttention(config)
            self.lang_inter = BertIntermediate(config)
            self.lang_output = BertOutput(config)

        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

        # The cross attention layer
        self.visual_attention = BertXAttention(config)
    
    def forward(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
        graph_sprels=None
    ):  
        '''
        1 step:  q = vision, k,v = lang, vision cross attend to lang
          ===> final output input + cross_attend_out (input_vision + vision_cross_lang)
        2 step : vision self attend (output of the first step)
        '''
        visn_att_output = self.visual_attention(
            visn_feats, lang_feats, ctx_att_mask=lang_attention_mask
        )[0]
        if graph_sprels is not None:
            visn_attention_mask = visn_attention_mask + graph_sprels
        visn_att_output = self.visn_self_att(visn_att_output, visn_attention_mask)[0]
        
        visn_inter_output = self.visn_inter(visn_att_output)
        visn_output = self.visn_output(visn_inter_output, visn_att_output)
        return visn_output
    
    def forward_lang2visn(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
    ):  
        '''
        1 step: q=lang, k,v=vision, lang cross attend to vision 
          ===> final output input + cross_attend_out  ( input_lang + lang_cross_vision)
        2 step: lang self attend (output of the first step)
        
        '''
        lang_att_output = self.visual_attention(
            lang_feats, visn_feats, ctx_att_mask=visn_attention_mask
        )[0]

        lang_att_output = self.lang_self_att(
            lang_att_output, lang_attention_mask
        )[0]
        lang_inter_output = self.lang_inter(lang_att_output)
        lang_output = self.lang_output(lang_inter_output, lang_att_output)
        return lang_output

class CrossmodalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_x_layers = config.num_x_layers
        self.x_layers = nn.ModuleList(
            [GraphLXRTXLayer(config) for _ in range(self.num_x_layers)]
        )

    def forward(self, txt_embeds, txt_masks, img_embeds, img_masks, graph_sprels=None):
        extended_txt_masks = None
        extended_img_masks = None
        if txt_masks != None:
            extended_txt_masks = extend_neg_masks(txt_masks)
        if img_masks != None:
            extended_img_masks = extend_neg_masks(img_masks)  # (N, 1(H), 1(L_q), L_v)
        for layer_module in self.x_layers:
            img_embeds = layer_module(
                txt_embeds, extended_txt_masks, 
                img_embeds, extended_img_masks,
                graph_sprels=graph_sprels
            )
        return img_embeds

class ImageEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.loc_linear = nn.Linear(config.angle_feat_size + 3, config.hidden_size)
        self.loc_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        if config.obj_feat_size > 0 and config.obj_feat_size != config.image_feat_size:
            self.obj_linear = nn.Linear(config.obj_feat_size, config.hidden_size)
            self.obj_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        else:
            self.obj_linear = self.obj_layer_norm = None

        # 0: non-navigable, 1: navigable, 2: object
        self.nav_type_embedding = nn.Embedding(3, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.num_pano_layers > 0:
            self.pano_encoder = create_transformer_encoder(
                config, config.num_pano_layers, norm=True
            )
        else:
            self.pano_encoder = None
    
    def forward(
        self, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, type_embed_layer
    ):  
        '''
        1. transforms img_feat and obj_feat
        2. cat [img_feat, obj_feat]
        '''
        device = traj_view_img_fts.device
        has_obj = traj_obj_img_fts is not None

        traj_view_img_embeds = self.img_layer_norm(self.img_linear(traj_view_img_fts))
        if has_obj:
            if self.obj_linear is None:
                traj_obj_img_embeds = self.img_layer_norm(self.img_linear(traj_obj_img_fts))
            else:
                traj_obj_img_embeds = self.obj_layer_norm(self.obj_linear(traj_obj_img_embeds))
            traj_img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                traj_view_img_embeds, traj_obj_img_embeds, traj_vp_view_lens, traj_vp_obj_lens
            ):
                if obj_len > 0:
                    traj_img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    traj_img_embeds.append(view_embed[:view_len])
            traj_img_embeds = pad_tensors_wgrad(traj_img_embeds)
            traj_vp_lens = traj_vp_view_lens + traj_vp_obj_lens
        else:
            traj_img_embeds = traj_view_img_embeds
            traj_vp_lens = traj_vp_view_lens

        traj_embeds = traj_img_embeds + \
                      self.loc_layer_norm(self.loc_linear(traj_loc_fts)) + \
                      self.nav_type_embedding(traj_nav_types) + \
                      type_embed_layer(torch.ones(1, 1).long().to(device))
        traj_embeds = self.layer_norm(traj_embeds)
        traj_embeds = self.dropout(traj_embeds)

        traj_masks = gen_seq_masks(traj_vp_lens)
        if self.pano_encoder is not None:
            traj_embeds = self.pano_encoder(
                traj_embeds, src_key_padding_mask=traj_masks.logical_not()
            )

        split_traj_embeds = torch.split(traj_embeds, traj_step_lens, 0)
        split_traj_vp_lens = torch.split(traj_vp_lens, traj_step_lens, 0)
        return split_traj_embeds, split_traj_vp_lens


class LocalVPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size*2 + 6, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.encoder = CrossmodalEncoder(config)

    def vp_input_embedding(self, split_traj_embeds, split_traj_vp_lens, vp_pos_fts):
        vp_img_embeds = pad_tensors_wgrad([x[-1] for x in split_traj_embeds])
        vp_lens = torch.stack([x[-1]+1 for x in split_traj_vp_lens], 0)
        vp_masks = gen_seq_masks(vp_lens)
        max_vp_len = max(vp_lens)

        batch_size, _, hidden_size = vp_img_embeds.size()
        device = vp_img_embeds.device
        # add [stop] token at beginning
        vp_img_embeds = torch.cat(
            [torch.zeros(batch_size, 1, hidden_size).to(device), vp_img_embeds], 1
        )[:, :max_vp_len]
        vp_embeds = vp_img_embeds + self.vp_pos_embeddings(vp_pos_fts)

        return vp_embeds, vp_masks
    
    def forward(
        self, txt_embeds, txt_masks, split_traj_embeds, split_traj_vp_lens, vp_pos_fts
    ):  
        '''
        1. step get viwe point embedding 
        2. cross attention (lang to view point embedding)
        '''
        vp_embeds, vp_masks = self.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        vp_embeds = self.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)
        return vp_embeds


class GlobalMapEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gmap_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.gmap_step_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        self.encoder = CrossmodalEncoder(config)
        
        if config.graph_sprels:
            self.sprel_linear = nn.Linear(1, 1)
        else:
            self.sprel_linear = None
    
    def _aggregate_gmap_features(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
    ):
        batch_size = len(split_traj_embeds)
        device = split_traj_embeds[0].device

        batch_gmap_img_fts = []
        for i in range(batch_size):
            visited_vp_fts, unvisited_vp_fts = {}, {}
            vp_masks = gen_seq_masks(split_traj_vp_lens[i])
            max_vp_len = max(split_traj_vp_lens[i])
            i_traj_embeds = split_traj_embeds[i][:, :max_vp_len] * vp_masks.unsqueeze(2)
            for t in range(len(split_traj_embeds[i])):
                visited_vp_fts[traj_vpids[i][t]] = torch.sum(i_traj_embeds[t], 0) / split_traj_vp_lens[i][t]
                for j, vp in enumerate(traj_cand_vpids[i][t]):
                    if vp not in visited_vp_fts:
                        unvisited_vp_fts.setdefault(vp, [])
                        unvisited_vp_fts[vp].append(i_traj_embeds[t][j])

            gmap_img_fts = []
            for vp in gmap_vpids[i][1:]:
                if vp in visited_vp_fts:
                    gmap_img_fts.append(visited_vp_fts[vp])
                else:
                    gmap_img_fts.append(torch.mean(torch.stack(unvisited_vp_fts[vp], 0), 0))
            gmap_img_fts = torch.stack(gmap_img_fts, 0)
            batch_gmap_img_fts.append(gmap_img_fts)

        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        # add a [stop] token at beginning
        batch_gmap_img_fts = torch.cat(
            [torch.zeros(batch_size, 1, batch_gmap_img_fts.size(2)).to(device), batch_gmap_img_fts], 
            dim=1
        )
        return batch_gmap_img_fts
    
    def gmap_input_embedding(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens
    ):
        gmap_img_fts = self._aggregate_gmap_features(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
        )
        gmap_embeds = gmap_img_fts + \
                      self.gmap_step_embeddings(gmap_step_ids) + \
                      self.gmap_pos_embeddings(gmap_pos_fts)
        gmap_masks = gen_seq_masks(gmap_lens)
        return gmap_embeds, gmap_masks

    def forward(
        self, txt_embeds, txt_masks,
        split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens, graph_sprels=None
    ):  

   
        gmap_embeds, gmap_masks = self.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens
        )
        
        if self.sprel_linear is not None:
            graph_sprels = self.sprel_linear(graph_sprels.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None

        gmap_embeds = self.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks,
            graph_sprels=graph_sprels
        )
        return gmap_embeds


class ClsPrediction(nn.Module):
    def __init__(self, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)


class RoomPrediction(nn.Module):
    def __init__(self, output_size, input_size, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            hidden_size = input_size 
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, output_size))
        

    def forward(self, x):
        return self.net(x)

class RoomPredictionImg(nn.Module):
    def __init__(self, config, output_size, input_size):
        super().__init__()
        self.room_type_list = []
        if config.rp_embed_dir is not None:
            rp_order = sorted(ROOM2IND.items(), key=lambda x: x[1])
            rp_embed_file = h5py.File(config.rp_embed_dir,"r")
            for r in rp_order:
                if config.features == 'clip':
                    rp_embed = rp_embed_file[r[0]+'_clip'][...][:, :config.image_feat_size]
                elif config.features == 'vitbase':
                    rp_embed = rp_embed_file[r[0]+'_imgnet_feat'][...][:, :config.image_feat_size]
                    if len(rp_embed.shape) == 4:
                        rp_embed = np.squeeze(rp_embed)
                rp_img_tensor = torch.from_numpy(rp_embed)
                linear = nn.Linear(input_size, output_size).cuda()
                
                linear.weight.data.copy_(rp_img_tensor.cuda())
                self.room_type_list.append(linear)
           
        if not config.update_rp_embed:
            for layer in self.room_type_list:
                for para in layer.parameters():
                    para.requires_grad = False
        
    def forward(self, view_feat):
        outs = []
        for layer in self.room_type_list:
            outs.append(torch.sum(layer(view_feat),dim=-1).unsqueeze(-1))
        outs = torch.cat(outs,dim=-1)
        return outs 


class RoomPrototypeEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.features_type = config.features

        if self.features_type == 'clip':
            self.rp_embed_layer = nn.Linear(30, config.image_feat_size)
        elif self.features_type == 'vitbase':
            self.rp_embed_layer = nn.Linear(30, 2048)
            self.feat_linear = nn.Linear(2048,768)

        if config.rp_embed_dir is not None:
            rp_embed_file = h5py.File(config.rp_embed_dir,"r")
            rp_order = sorted(ROOM2IND.items(), key=lambda x: x[1])
            rp_img_embeds = []
            for r in rp_order:
                if config.features == 'clip':
                    rp_embed = rp_embed_file[r[0]+'_clip'][...].mean(axis=0)
                elif config.features == 'vitbase':
                    rp_embed = rp_embed_file[r[0]+'_imgnet_feat'][...].mean(axis=0)
                rp_img_embeds.append(rp_embed)
            rp_embed_file.close()

            rp_img_tensor = torch.from_numpy(np.stack(rp_img_embeds)).squeeze(-1).squeeze(-1)
            self.rp_embed_layer.weight.data.copy_(rp_img_tensor.permute(1,0))
            if not config.update_rp_embed:
                for para in self.rp_embed_layer.parameters():
                    para.requires_grad = False

    def forward(self, layout_logits):

        rp_embeds = self.rp_embed_layer(layout_logits)
        if self.features_type == 'vitbase':
            rp_embeds = self.feat_linear(rp_embeds)
        return rp_embeds

from .node_dist_module import NodeDistReg, NodeVisReg
class GlocalTextPathNavCMT(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.lang_encoder = LanguageEncoder(config)  # 9 layer bert

        self.img_embeddings = ImageEmbeddings(config)
        
        self.local_encoder = LocalVPEncoder(config)
        self.global_encoder = GlobalMapEncoder(config)
        self.global_visual_encoder = GlobalVisualEncoder(config)

        self.global_sap_head = ClsPrediction(self.config.hidden_size)
        self.local_sap_head = ClsPrediction(self.config.hidden_size)

        if not config.use_img_room_head:
            self.global_room_cls_head = RoomPrediction(output_size=30,
                                                   input_size=self.config.hidden_size)
        else:
            self.global_room_cls_head = RoomPredictionImg(config=config, output_size=10, input_size=config.hidden_size)
        
        if config.use_gd:
            self.node_dis_reg_head = NodeDistReg(input_size=self.config.hidden_size, config=config)
            self.local_rec_reg_head = NodeVisReg(input_size=self.config.hidden_size, config=config)
            self.global_distsap_head = ClsPrediction(self.config.hidden_size)
            self.local_rec_sap_head = ClsPrediction(self.config.hidden_size)
                                    
        self.global_fuse_linear = None 

        if config.h_graph:
            self.layout_embed_layer = RoomPrototypeEmbedding(config)
            self.layout_attention = LayoutGraphAttention(config)
            self.layout_x = LayoutGrpahX(config)
            self.layout_sap_head = ClsPrediction(self.config.hidden_size)
            
            if config.global_fuse:
                self.global_fuse_linear = ClsPrediction(self.config.hidden_size, input_size = self.config.hidden_size * 2)

        if config.glocal_fuse:
            self.sap_fuse_linear = ClsPrediction(self.config.hidden_size, input_size = self.config.hidden_size * 2)
        else:
            self.sap_fuse_linear = None 
        
        if self.config.fuse_dist_score_to_global:
            #self.fuse_dist_weight_global = self.config.fuse_dist_score_to_global
            self.sap_global_fuse_linear = ClsPrediction(self.config.hidden_size, input_size = self.config.hidden_size*2)
        
        if self.config.obj_feat_size > 0:
            self.og_head = ClsPrediction(self.config.hidden_size)

        self.knowledge_proj = nn.Linear(512, 768)
        self.crop_proj = nn.Linear(512, 768)
        self.instruction_proj = nn.Linear(768, 768)
        self.history_proj = nn.Linear(768, 768)
        self.fusion_proj = nn.Linear(768, 768)

        knowledge_config = copy.deepcopy(config)
        knowledge_config.num_x_layers = 2

        self.cross_vision_knowledge = CrossmodalEncoder(knowledge_config)
        self.cross_history_knowledge = CrossmodalEncoder(knowledge_config)

        self.fusion_layer_norm = BertLayerNorm(768, eps=1e-12)

        self.final_ffn = nn.Linear(768 * 2, 768)
        self.final_layer_norm = BertLayerNorm(768, eps=1e-12)
        scale = 768 ** -0.5
        self.no_history_embedding = nn.Parameter(scale * torch.randn(1, 768))
        
        self.init_weights()

        if config.fix_lang_embedding or config.fix_local_branch:
            for k, v in self.embeddings.named_parameters():
                v.requires_grad = False
            for k, v in self.lang_encoder.named_parameters():
                v.requires_grad = False

        if config.fix_pano_embedding or config.fix_local_branch:
            for k, v in self.img_embeddings.named_parameters():
                v.requires_grad = False

        if config.fix_local_branch:
            for k, v in self.local_encoder.named_parameters():
                v.requires_grad = False
            for k, v in self.local_sap_head.named_parameters():
                v.requires_grad = False
            for k, v in self.og_head.named_parameters():
                v.requires_grad = False
        

    def forward_text(self, txt_ids, txt_masks, ins2imgs=None):
        batch_size = txt_ids.shape[0]
        txt_token_type_ids = torch.zeros_like(txt_ids)
        if ins2imgs is not None:
            num_img = ins2imgs.shape[1]
            txt_length = torch.sum(txt_masks,dim=1) 
            end_inds = []
            for i in range(batch_size):
                t_len = txt_length[i].item()
                end_txt_ind = t_len - num_img - 1
                end_inds.append(end_txt_ind)
                txt_token_type_ids[i,end_txt_ind:end_txt_ind+num_img+1] = 1
            txt_embeds = self.embeddings(txt_ids, token_type_ids = txt_token_type_ids)
            for i in range(batch_size):
                txt_embeds[i, end_inds[i]:end_inds[i]+num_img] = torch.from_numpy(ins2imgs[i])
        else:
            txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
        txt_embeds = self.lang_encoder(txt_embeds, txt_masks)
        return txt_embeds
    

    def forward_panorama_per_step(
        self, view_img_fts, rec_view_img_fts, obj_img_fts, loc_fts, nav_types, view_lens, obj_lens,
            instruction_fts, knowledge_fts, crop_fts, used_cand_ids, gmap_img_embeds, gmap_step_ids, gmap_pos_fts
    ):
        global CROP_SIZE
        batch_size = view_img_fts.size(0)

        # History feature
        if gmap_img_embeds != None:
            gmap_embeds = gmap_img_embeds + \
                          self.global_encoder.gmap_step_embeddings(gmap_step_ids) + \
                          self.global_encoder.gmap_pos_embeddings(gmap_pos_fts)

            history_fts = self.history_proj(gmap_embeds).unsqueeze(1).repeat(1, 36, 1, 1)
            history_fts_shape = history_fts.shape

        knowledge_fts = self.knowledge_proj(knowledge_fts)
        knowledge_fts_shape = knowledge_fts.shape
        crop_fts = self.crop_proj(crop_fts)
        crop_fts_shape = crop_fts.shape

        instruction_fts_pure = self.instruction_proj(instruction_fts).permute(0, 2, 1).unsqueeze(1).repeat(1, 36, 1, 1)

        # Purification
        if gmap_img_embeds != None:
            history_matrix, _ = torch.matmul(history_fts, instruction_fts_pure).max(dim=-1)

        crop_matrix, _ = torch.matmul(crop_fts, instruction_fts_pure).max(dim=-1)

        knowledge_matrix, _ = torch.matmul(knowledge_fts, instruction_fts_pure).max(dim=-1)

        if gmap_img_embeds != None:
            history_purify_weight = torch.softmax(history_matrix / math.sqrt(768), dim=-1)

        crop_purify_weight = torch.softmax(crop_matrix / math.sqrt(768), dim=-1)

        knowledge_purify_weight = torch.softmax(knowledge_matrix / math.sqrt(768), dim=-1)

        if gmap_img_embeds != None:
            history_fts = history_fts * history_purify_weight.unsqueeze(-1)
        crop_fts = crop_fts * crop_purify_weight.unsqueeze(-1)
        knowledge_fts = knowledge_fts * knowledge_purify_weight.unsqueeze(-1)

        # Interaction
        crop_knowledge_fts = self.cross_vision_knowledge(knowledge_fts.view(batch_size * 36, 1, 768), None,
                                                         crop_fts.view(batch_size * 36, CROP_SIZE, 768), None).view(
            batch_size, 36, CROP_SIZE, 768)

        if gmap_img_embeds != None:
            history_knowledge_fts = self.cross_history_knowledge(
                knowledge_fts.view(batch_size * 36, 1, 768), None,
                history_fts.view(batch_size * 36, -1, 768), None).view(batch_size, 36, -1, 768)

        # Fusion
        instruction_cls = self.fusion_proj(instruction_fts[:, 0:1, :]).permute(0, 2, 1).unsqueeze(1).repeat(1, 36, 1, 1)

        if gmap_img_embeds != None:
            history_knowledge_logits = torch.matmul(history_knowledge_fts, instruction_cls) / math.sqrt(768)
            history_knowledge_weight = torch.softmax(history_knowledge_logits, dim=-2)
            history_knowledge_fts = (history_knowledge_fts * history_knowledge_weight).sum(-2)
            history_knowledge_fts = self.fusion_layer_norm(history_knowledge_fts)

        crop_knowledge_logits = torch.matmul(crop_knowledge_fts, instruction_cls) / math.sqrt(768)
        crop_knowledge_weight = torch.softmax(crop_knowledge_logits, dim=-2)
        crop_knowledge_fts = (crop_knowledge_fts * crop_knowledge_weight).sum(-2)
        crop_knowledge_fts = self.fusion_layer_norm(crop_knowledge_fts)

        device = view_img_fts.device
        device2 = rec_view_img_fts.device
        has_obj = obj_img_fts is not None

        # Fuse pano_fts, crop_knowledge_fts, history_knowledge_fts

        if gmap_img_embeds != None:
            fusion_fts = torch.cat([crop_knowledge_fts, history_knowledge_fts], dim=-1)

        else:
            fusion_fts = torch.cat(
                [crop_knowledge_fts, self.no_history_embedding.unsqueeze(0).repeat(batch_size, 36, 1)], dim=-1)

        fusion_fts = self.final_ffn(fusion_fts)
        fusion_fts = self.final_layer_norm(fusion_fts)

        for i in range(batch_size):
            view_ft_id = 0
            for cand_id in used_cand_ids[i]:
                view_img_fts[i, view_ft_id] = fusion_fts[i, cand_id] + view_img_fts[i, view_ft_id]
                rec_view_img_fts[i, view_ft_id] = fusion_fts[i, cand_id] + rec_view_img_fts[i, view_ft_id]
                view_ft_id += 1

            for cand_id in range(36):
                if cand_id not in used_cand_ids[i]:
                    view_img_fts[i, view_ft_id] = fusion_fts[i, cand_id] + view_img_fts[i, view_ft_id]
                    rec_view_img_fts[i, view_ft_id] = fusion_fts[i, cand_id] + rec_view_img_fts[i, view_ft_id]
                    view_ft_id += 1

        # caption-instruction enhanced
        view_img_embeds = self.img_embeddings.img_layer_norm(
            self.img_embeddings.img_linear(view_img_fts)
        )
        rec_view_img_embeds = self.img_embeddings.img_layer_norm(
            self.img_embeddings.img_linear(rec_view_img_fts)
        )
        if has_obj:
            
            if self.img_embeddings.obj_linear is None:
                obj_img_embeds = self.img_embeddings.img_layer_norm(
                    self.img_embeddings.img_linear(obj_img_fts)
                )
            else:
                obj_img_embeds = self.img_embeddings.obj_layer_norm(
                    self.img_embeddings.obj_linear(obj_img_fts)
                )
            img_embeds = []
            rec_img_embeds = []
            for view_embed, rec_view_embed, obj_embed, view_len, obj_len in zip(
                view_img_embeds, rec_view_img_embeds, obj_img_embeds, view_lens, obj_lens
            ):  
                # cat obj feature to the back of view feature 
                if obj_len > 0:
                    img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]],0))
                    rec_img_embeds.append(torch.cat([rec_view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    img_embeds.append(view_embed[:view_len])
                    rec_img_embeds.append(rec_view_embed[:view_len])
            img_embeds = pad_tensors_wgrad(img_embeds) ## [ [view features], [obj features] ]
            rec_img_embeds = pad_tensors_wgrad(rec_img_embeds)
            pano_lens = view_lens + obj_lens
        else:
            img_embeds = view_img_embeds
            rec_img_embeds = rec_view_img_embeds
            pano_lens = view_lens 
        
        pano_embeds = img_embeds + \
                      self.img_embeddings.loc_layer_norm(self.img_embeddings.loc_linear(loc_fts)) + \
                      self.img_embeddings.nav_type_embedding(nav_types) + \
                      self.embeddings.token_type_embeddings(torch.ones(1, 1).long().to(device))
        pano_embeds = self.img_embeddings.layer_norm(pano_embeds)
        pano_embeds = self.img_embeddings.dropout(pano_embeds)

        rec_pano_embeds = rec_img_embeds + \
                      self.img_embeddings.loc_layer_norm(self.img_embeddings.loc_linear(loc_fts)) + \
                      self.img_embeddings.nav_type_embedding(nav_types) + \
                      self.embeddings.token_type_embeddings(torch.ones(1, 1).long().to(device2))
        rec_pano_embeds = self.img_embeddings.layer_norm(rec_pano_embeds)
        rec_pano_embeds = self.img_embeddings.dropout(rec_pano_embeds)

        pano_masks = gen_seq_masks(pano_lens)
        rec_pano_masks = gen_seq_masks(pano_lens)
        if self.img_embeddings.pano_encoder is not None:
            pano_embeds = self.img_embeddings.pano_encoder(
                pano_embeds, src_key_padding_mask=pano_masks.logical_not()
            )
            rec_pano_embeds = self.img_embeddings.pano_encoder(
                rec_pano_embeds, src_key_padding_mask=rec_pano_masks.logical_not()
            )
        
        return pano_embeds, rec_pano_embeds, pano_masks, rec_pano_masks
    
    def forward_navigation_per_step(
        self, txt_embeds, txt_masks, gmap_img_embeds, gmap_step_ids, gmap_pos_fts,
        gmap_masks, gmap_pair_dists, gmap_visited_masks, gmap_vpids,
        vp_img_embeds, vp_pos_fts, vp_masks, vp_nav_masks, vp_obj_masks, vp_cand_vpids,
    ):  
        # vp_img_embeds[0] --> zero vector --> stop action 
        # gmap_img_embeds[0] --> zero vector --> stop action
        batch_size = txt_embeds.size(0)
        # global branch 
        gmap_embeds = gmap_img_embeds + \
                      self.global_encoder.gmap_step_embeddings(gmap_step_ids) + \
                      self.global_encoder.gmap_pos_embeddings(gmap_pos_fts)
        
        if self.global_encoder.sprel_linear is not None:
            graph_sprels = self.global_encoder.sprel_linear(
                gmap_pair_dists.unsqueeze(3)
            ).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None 

        gmap_embeds = self.global_encoder.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks,
            graph_sprels = graph_sprels
        )

        # local branch
        vp_embeds = vp_img_embeds + self.local_encoder.vp_pos_embeddings(vp_pos_fts)
        vp_embeds = self.local_encoder.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)
         # -> same operation as global 


        # navigation logits
        if self.sap_fuse_linear is None:
            fuse_weights = 0.5 
        else:
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))
        
        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(gmap_visited_masks, -float('inf')) # mask visited node
        global_logits.masked_fill_(gmap_masks.logical_not(), -float('inf')) # mask padding
        
        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1-fuse_weights)
        local_logits.masked_fill_(vp_nav_masks.logical_not(), -float('inf')) # masked padding
        
        # fusion 
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0] # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp,mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask ])
            tmp = {}
            bw_logits = 0

            for j, cand_vpid in enumerate(vp_cand_vpids[i]): # handle local branch 
                # 0 for stop so skip 
                if j > 0:
                    if cand_vpid in visited_nodes:
                        bw_logits += local_logits[i, j]
                    else:
                        tmp[cand_vpid] = local_logits[i, j]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes: # unvisited node in gmap
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:  # this vp also no in local vp_cand_vpids
                        fused_logits[i, j] += bw_logits

        # object grounding logits
     
        if vp_obj_masks is not None:
            obj_logits = self.og_head(vp_embeds).squeeze(2)
            obj_logits.masked_fill_(vp_obj_masks.logical_not(), -float('inf'))
        else:
            obj_logits = None 
   
        outs = {
            'gmap_embeds': gmap_embeds,
            'vp_embeds': vp_embeds,
            'global_logits': global_logits,
            'local_logits': local_logits,
            'fused_logits': fused_logits,
            'obj_logits': obj_logits,
        }

        return outs 

    def forward_navigation_with_room_type(
        self, txt_embeds, txt_masks, gmap_img_embeds, gmap_step_ids, gmap_pos_fts,
        gmap_masks, gmap_pair_dists, gmap_visited_masks, gmap_vpids, 
        vp_img_embeds, vp_pos_fts, vp_masks, vp_nav_masks, vp_obj_masks, vp_cand_vpids,
    ):  
        batch_size = txt_embeds.size(0)

        # global branch
        gmap_embeds = gmap_img_embeds + \
                      self.global_encoder.gmap_step_embeddings(gmap_step_ids) + \
                      self.global_encoder.gmap_pos_embeddings(gmap_pos_fts)

        if self.global_encoder.sprel_linear is not None:
            graph_sprels = self.global_encoder.sprel_linear(
                gmap_pair_dists.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None 
        
        
        gmap_embeds = self.global_visual_encoder(gmap_embeds,gmap_masks,graph_sprels)
        #gmap_room_type = self.global_room_cls_head(gmap_embeds)

        gmap_embeds = self.global_encoder.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks,
            graph_sprels = graph_sprels
        )

        gmap_room_type = self.global_room_cls_head(gmap_embeds)
        # local_branch
        vp_embeds = vp_img_embeds + self.local_encoder.vp_pos_embeddings(vp_pos_fts)
        vp_embeds = self.local_encoder.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)
 
        # navigation logits
        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else:
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))
        # print(fuse_weights)

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gmap_masks.logical_not(), -float('inf'))
        # print('global', torch.softmax(global_logits, 1)[0], global_logits[0])

        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        local_logits.masked_fill_(vp_nav_masks.logical_not(), -float('inf'))
        # print('local', torch.softmax(local_logits, 1)[0], local_logits[0])

        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]   # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(vp_cand_vpids[i]):
                if j > 0:
                    if cand_vpid in visited_nodes:
                        bw_logits += local_logits[i, j]
                    else:
                        tmp[cand_vpid] = local_logits[i, j]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits
        # print('fused', torch.softmax(fused_logits, 1)[0], fused_logits[0])

        # object grounding logits
        if vp_obj_masks is not None:
            obj_logits = self.og_head(vp_embeds).squeeze(2)
            obj_logits.masked_fill_(vp_obj_masks.logical_not(), -float('inf'))
        else:
            obj_logits = None

        outs = {
            'gmap_embeds': gmap_embeds,
            'vp_embeds': vp_embeds,
            'global_logits': global_logits,
            'local_logits': local_logits,
            'fused_logits': fused_logits,
            'obj_logits': obj_logits,
            'room_type_pred': gmap_room_type
        }
        return outs
    
    def forward_navigation_with_room_type_node_dist(
        self, txt_embeds, txt_masks, gmap_img_embeds, gmap_step_ids, gmap_pos_fts,
        gmap_masks, gmap_pair_dists, gmap_visited_masks, gmap_vpids, 
        vp_img_embeds, vp_pos_fts, vp_masks, vp_nav_masks, vp_obj_masks, vp_cand_vpids,
        ins2img, curr_vid_idx
    ):  
        batch_size = txt_embeds.size(0)

        # global branch
        gmap_embeds = gmap_img_embeds + \
                      self.global_encoder.gmap_step_embeddings(gmap_step_ids) + \
                      self.global_encoder.gmap_pos_embeddings(gmap_pos_fts)

        if self.global_encoder.sprel_linear is not None:
            graph_sprels = self.global_encoder.sprel_linear(
                gmap_pair_dists.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None 
        
        
        gmap_embeds = self.global_visual_encoder(gmap_embeds,gmap_masks,graph_sprels)

        gmap_embeds = self.global_encoder.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks,
            graph_sprels = graph_sprels
        )

        gmap_room_type = self.global_room_cls_head(gmap_embeds)

        _, dist_logits, fuse_weight2 = self.node_dis_reg_head(gmap_embeds, ins2img)

        # local_branch
        vp_embeds = vp_img_embeds + self.local_encoder.vp_pos_embeddings(vp_pos_fts)
        vp_embeds = self.local_encoder.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)
 
        # navigation logits
        fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))
        # print(fuse_weights)
        if self.config.const_fuse_gl:
            fuse_weights = self.config.const_fuse_gl_weight
        
        if self.config.const_fuse_gd:
            fuse_weight2 = self.config.const_fuse_gd_weight

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        if self.config.switch_first_gd:
            global_logits[:,0] = global_logits[np.arange(len(curr_vid_idx)), np.array(curr_vid_idx)]
        # if self.config.bw_weight:
        #     global_logits = global_logits * self.config.bw_weight
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gmap_masks.logical_not(), -float('inf'))
        # print('global', torch.softmax(global_logits, 1)[0], global_logits[0])

        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        local_logits.masked_fill_(vp_nav_masks.logical_not(), -float('inf'))
        # print('local', torch.softmax(local_logits, 1)[0], local_logits[0])

        fused_logits2 = self.global_distsap_head(gmap_embeds).squeeze(2)
        if self.config.switch_first_gd:
            dist_logits[:,0] = dist_logits[np.arange(len(curr_vid_idx)), np.array(curr_vid_idx)]
            fused_logits2[:,0] = fused_logits2[np.arange(len(curr_vid_idx)), np.array(curr_vid_idx)]
        # if self.config.bw_weight:
        #     fused_logits2 = fused_logits2 * self.config.bw_weight

        fused_logits2 = dist_logits * (1-fuse_weight2) + fused_logits2 * fuse_weight2
        fused_logits2.masked_fill_(gmap_visited_masks, -float('inf'))
        fused_logits2.masked_fill_(gmap_masks.logical_not(), -float('inf')) 

        dist_logits_copy = torch.clone(dist_logits)
        dist_logits_copy.masked_fill_(gmap_visited_masks, -float('inf'))
        dist_logits_copy.masked_fill_(gmap_masks.logical_not(), -float('inf')) 

        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]   # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            n_accum = 0
            for j, cand_vpid in enumerate(vp_cand_vpids[i]):
                if j > 0:
                    if cand_vpid in visited_nodes:
                        bw_logits += local_logits[i, j]
                        n_accum += 1
                    else:
                        tmp[cand_vpid] = local_logits[i, j]

            if self.config.bw_weight and bw_logits>0:
                bw_logits = bw_logits * self.config.bw_weight

            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        # if n_accum != 0 and bw_logits>0 and self.config.avg_local_emb:
                        if n_accum != 0 and self.config.avg_local_emb:
                            fused_logits[i, j] += bw_logits/n_accum
                        else:
                            fused_logits[i, j] += bw_logits
        # print('fused', torch.softmax(fused_logits, 1)[0], fused_logits[0])

        fused_logits += fused_logits2
        # object grounding logits
        if vp_obj_masks is not None:
            obj_logits = self.og_head(vp_embeds).squeeze(2)
            obj_logits.masked_fill_(vp_obj_masks.logical_not(), -float('inf'))
        else:
            obj_logits = None

        outs = {
            'gmap_embeds': gmap_embeds,
            'vp_embeds': vp_embeds,
            'global_logits': global_logits,
            'local_logits': local_logits,
            'fused_logits': fused_logits,
            'obj_logits': obj_logits,
            'room_type_pred': gmap_room_type,
            'dist_logits': dist_logits_copy,
        }
        return outs
    
    def forward_navigation_with_rt_gd(
        self, txt_embeds, txt_masks, gmap_img_embeds, gmap_step_ids, gmap_pos_fts,
        gmap_masks, gmap_pair_dists, gmap_visited_masks, gmap_vpids, 
        vp_img_embeds, vp_rec_img_embeds, vp_pos_fts, vp_masks, vp_nav_masks, vp_obj_masks, vp_cand_vpids,
        ins2img, curr_vid_idx
    ):  
        batch_size = txt_embeds.size(0)

        # global branch
        gmap_embeds = gmap_img_embeds + \
                      self.global_encoder.gmap_step_embeddings(gmap_step_ids) + \
                      self.global_encoder.gmap_pos_embeddings(gmap_pos_fts)

        if self.global_encoder.sprel_linear is not None:
            graph_sprels = self.global_encoder.sprel_linear(
                gmap_pair_dists.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None 
        
        
        gmap_embeds = self.global_visual_encoder(gmap_embeds,gmap_masks,graph_sprels)

        gmap_embeds = self.global_encoder.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks,
            graph_sprels = graph_sprels
        )

        gmap_room_type = self.global_room_cls_head(gmap_embeds)
        _, dist_logits, fuse_weight2 = self.node_dis_reg_head(gmap_embeds, ins2img)

        # local_branch
        vp_embeds = vp_img_embeds + self.local_encoder.vp_pos_embeddings(vp_pos_fts)
        vp_embeds = self.local_encoder.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)

        vp_rec_embeds = vp_rec_img_embeds + self.local_encoder.vp_pos_embeddings(vp_pos_fts)
        vp_rec_embeds = self.local_encoder.encoder(txt_embeds, txt_masks, vp_rec_embeds, vp_masks)
        _, rec_logits, fuse_weight3 = self.local_rec_reg_head(vp_embeds, vp_rec_embeds)
 
        # fuse_weights = 0.5
        fuse_weights = torch.sigmoid(self.sap_fuse_linear(
            torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
        ))

        # global
        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gmap_masks.logical_not(), -float('inf'))

        # local
        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        local_logits.masked_fill_(vp_nav_masks.logical_not(), -float('inf'))

        # gd
        fused_logits2 = self.global_distsap_head(gmap_embeds).squeeze(2)

        fused_logits2 = dist_logits * (1-fuse_weight2) + fused_logits2 * fuse_weight2
        fused_logits2.masked_fill_(gmap_visited_masks, -float('inf'))
        fused_logits2.masked_fill_(gmap_masks.logical_not(), -float('inf'))

        # visionary
        fused_logits3 = self.local_rec_sap_head(vp_embeds).squeeze(2)
        fused_logits3 = rec_logits * (1 - fuse_weight3) + fused_logits3 * fuse_weight3
        fused_logits3.masked_fill_(vp_nav_masks.logical_not(), -float('inf'))

        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]   # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            n_accum = 0
            for j, cand_vpid in enumerate(vp_cand_vpids[i]):
                if j > 0:
                    if cand_vpid in visited_nodes:
                        bw_logits += local_logits[i, j]
                        n_accum += 1
                    else:
                        tmp[cand_vpid] = local_logits[i, j]

            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits

        fused_rec_logits = torch.clone(global_logits)
        fused_rec_logits[:, 0] += fused_logits3[:, 0]  # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            n_accum = 0
            for j, cand_vpid in enumerate(vp_cand_vpids[i]):
                if j > 0:
                    if cand_vpid in visited_nodes:
                        bw_logits += fused_logits3[i, j]
                        n_accum += 1
                    else:
                        tmp[cand_vpid] = fused_logits3[i, j]

            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_rec_logits[i, j] += tmp[vp]
                    else:
                        fused_rec_logits[i, j] += bw_logits

        # add gd
        fused_logits += fused_logits2
        # add rec_vis
        fused_logits += fused_rec_logits

        # object grounding logits
        if vp_obj_masks is not None:
            obj_logits = self.og_head(vp_embeds).squeeze(2)
            obj_logits.masked_fill_(vp_obj_masks.logical_not(), -float('inf'))
        else:
            obj_logits = None

        outs = {
            'gmap_embeds': gmap_embeds,
            'vp_embeds': vp_embeds,
            'global_logits': global_logits,
            'local_logits': local_logits,
            'fused_logits': fused_logits,
            'obj_logits': obj_logits,
            'room_type_pred': gmap_room_type,
        }
        return outs

    def forward(self, mode, batch, **kwargs):
        if mode == 'language':
            txt_embeds = self.forward_text(batch['txt_ids'], batch['txt_masks'], batch['ins2img'])
            return txt_embeds
        
        elif mode == 'panorama':
            pano_embeds, rec_pano_embeds, pano_masks, rec_pano_masks = self.forward_panorama_per_step(
                batch['view_img_fts'], batch['rec_view_img_fts'], batch['obj_img_fts'], batch['loc_fts'],
                batch['nav_types'], batch['view_lens'], batch['obj_lens'],
                batch['instruction_fts'], batch['knowledge_fts'], batch['crop_fts'], batch['used_cand_ids'],
                batch['gmap_img_embeds'], batch['gmap_step_ids'], batch['gmap_pos_fts']
            )

            return pano_embeds, rec_pano_embeds, pano_masks, rec_pano_masks
        
        elif mode == 'navigation':
            return self.forward_navigation_per_step(
                batch['txt_embeds'], batch['txt_masks'], batch['gmap_img_embeds'],
                batch['gmap_step_ids'], batch['gmap_pos_fts'], batch['gmap_masks'],
                batch['gmap_pair_dists'], batch['gmap_visited_masks'], batch['gmap_vpids'],
                batch['vp_img_embeds'], batch['vp_pos_fts'], batch['vp_masks'],
                batch['vp_nav_masks'], batch['vp_obj_masks'], batch['vp_cand_vpids'],
            )
        elif mode == 'navigation_with_room_type':
            return self.forward_navigation_with_room_type(
                batch['txt_embeds'],batch['txt_masks'],batch['gmap_img_embeds'],
                batch['gmap_step_ids'],batch['gmap_pos_fts'],batch['gmap_masks'],
                batch['gmap_pair_dists'], batch['gmap_visited_masks'],batch['gmap_vpids'],
                batch['vp_img_embeds'], batch['vp_pos_fts'], batch['vp_masks'],
                batch['vp_nav_masks'], batch['vp_obj_masks'], batch['vp_cand_vpids'],
            )
        elif mode == 'navigation_with_room_type_node_dist': # tuning version with multiple tuning config
            return self.forward_navigation_with_room_type_node_dist(
                batch['txt_embeds'],batch['txt_masks'],batch['gmap_img_embeds'],
                batch['gmap_step_ids'],batch['gmap_pos_fts'],batch['gmap_masks'],
                batch['gmap_pair_dists'], batch['gmap_visited_masks'],batch['gmap_vpids'],
                batch['vp_img_embeds'], batch['vp_pos_fts'], batch['vp_masks'],
                batch['vp_nav_masks'], batch['vp_obj_masks'], batch['vp_cand_vpids'],
                batch['ins2img'], batch['curr_vid_idx']
            )
        elif mode == 'navigation_with_rt_gd':  # stable version -- cur
            return self.forward_navigation_with_rt_gd(
                batch['txt_embeds'], batch['txt_masks'], batch['gmap_img_embeds'],
                batch['gmap_step_ids'], batch['gmap_pos_fts'], batch['gmap_masks'],
                batch['gmap_pair_dists'], batch['gmap_visited_masks'], batch['gmap_vpids'],
                batch['vp_img_embeds'], batch['vp_rec_img_embeds'], batch['vp_pos_fts'], batch['vp_masks'],
                batch['vp_nav_masks'], batch['vp_obj_masks'], batch['vp_cand_vpids'],
                batch['ins2img'], batch['curr_vid_idx']
            )


