"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Misc lr helper
"""
from torch.optim import Adam, Adamax

from .adamw import AdamW
from .rangerlars import RangerLars

def build_optimizer(model, opts):
    param_optimizer = list(model.named_parameters())

    update_param_name = ['knowledge_proj', 'crop_proj', 'instruction_proj', 'history_proj', 'fusion_proj',
                         'cross_vision_knowledge', 'cross_history_knowledge', 'fusion_layer_norm', 'final_ffn',
                         'no_history_embedding', 'final_layer_norm', 'local_rec_sap_head', 'local_rec_reg_head']

    update_param = [(n, p) for n, p in param_optimizer if any(nd in n for nd in update_param_name)]
    freeze_param = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in update_param_name)]

    print('update_param', [n for n, p in update_param])
    print('\n\n')
    print('freeze_param', [n for n, p in freeze_param])

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in update_param if not any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,  # 训练
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in update_param if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in freeze_param if not any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate * 1e-3,  # 微调
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in freeze_param if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    elif opts.optim == 'rangerlars':
        OptimCls = RangerLars
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    return optimizer

# def build_optimizer(model, opts):
#     param_optimizer = list(model.named_parameters())
#     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in param_optimizer
#                     if not any(nd in n for nd in no_decay)],
#          'weight_decay': opts.weight_decay},
#         {'params': [p for n, p in param_optimizer
#                     if any(nd in n for nd in no_decay)],
#          'weight_decay': 0.0}
#     ]
#
#     # currently Adam only
#     if opts.optim == 'adam':
#         OptimCls = Adam
#     elif opts.optim == 'adamax':
#         OptimCls = Adamax
#     elif opts.optim == 'adamw':
#         OptimCls = AdamW
#     elif opts.optim == 'rangerlars':
#         OptimCls = RangerLars
#     else:
#         raise ValueError('invalid optimizer')
#     optimizer = OptimCls(optimizer_grouped_parameters,
#                          lr=opts.learning_rate, betas=opts.betas)
#     return optimizer


def update_optimizer(model, opts, optimizer, training_modules = None, name_set = None):
    param_optimizer = list(model.named_parameters())
    if name_set is not None:
        param_optimizer = [pair for pair in param_optimizer if pair[0] not in name_set]

    param_optimzer_tmp = []
    if training_modules is not None:
        for training_module in training_modules:
            param_optimzer_tmp.append([pair for pair in param_optimizer if training_module in pair[0]])
        param_optimizer = sum(param_optimzer_tmp, [])      
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    for group in optimizer_grouped_parameters:
        optimizer.add_param_group(group)
    out_name_set = {n for n, p in param_optimizer}
    return out_name_set
    