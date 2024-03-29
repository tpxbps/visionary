DATA_ROOT=../datasets
timestamp=$(date +%m%d-%H%M%S)
OUT_ROOT=../out/REVERIE/experiments

train_alg=dagger
features=clip
ft_dim=768
obj_features=vitbase
obj_ft_dim=768
rp_img_dir=../room_type_feats.h5

ngpus=1
seed=0 # default 0

name=${train_alg}-${features}
name=${name}-seed.${seed} #-${ngpus}gpus

outdir=${OUT_ROOT}/reverie_finetune # path to save logs

#warmup_ckpt=../ckpts/model_step_8000.pt

flag="
  --root_dir ${DATA_ROOT}
  --dataset reverie
  --output_dir ${outdir}
  --world_size ${ngpus}
  --seed ${seed}
  --tokenizer bert

  --enc_full_graph
  --graph_sprels
  --fusion dynamic
  --multi_endpoints
  --use_room_type
  --use_img_room_head
  --dagger_sample sample

  --train_alg ${train_alg}
  
  --num_l_layers 9
  --num_x_layers 4
  --num_pano_layers 2
  --num_v_layers 4
  
  --max_action_len 15
  --max_instr_len 200
  --max_objects 20

  --batch_size 2
  --lr 1e-5
  --iters 200000
  --log_every 1000
  --optim adamW

  --features ${features}
  --obj_features ${obj_features}
  --image_feat_size ${ft_dim}
  --angle_feat_size 4
  --obj_feat_size ${obj_ft_dim}
  --rp_embed_dir ${rp_img_dir}

  --ml_weight 0.2
  --feat_dropout 0.4
  --dropout 0.5
  --gamma 0.
  
  --node_loss_delta 1.0
  --use_gd
  --stable_gd
  --use_real_dist_norm
  --num_of_ins_img 5
  --gd_dreamer_type attn_dynamic_fuse"

#  --bert_ckpt_file $warmup_ckpt \
PYTHONPATH=../:$PYTHONPATH python nav_obj.py $flag  \
  --tokenizer bert \
  --resume_file ../ckpts/lad_model_full.pt \
  --eval_first
