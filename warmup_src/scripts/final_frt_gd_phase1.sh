NODE_RANK=0
NUM_GPUS=1
DATA_ROOT=../datasets
outdir=../out/REVERIE/experiments/pretrain/frt_gd_phase1
rt_img_dir=../room_type_feats.h5

phase_ckpt=../ckpts/warmup_stage2.pt

# train
PYTHONPATH="../":$PYTHONPATH python3 train.py \
    --output_dir $outdir \
    --model_config config/reverie_obj_model_config.json \
    --config config/reverie_pretrain_rt_gd.json \
    --use_rt_task \
    --vlnbert cmt \
    --use_clip_feat \
    --rt_embed_dir $rt_img_dir \
    --use_fix_rt_emb \
    --init_ckpt $phase_ckpt \
    --start_from 1

#PYTHONPATH="../":$PYTHONPATH python -m torch.distributed.launch \
#    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
#    train.py --world_size ${NUM_GPUS} \
#    --output_dir $outdir \
#    --model_config config/reverie_obj_model_config.json \
#    --config config/reverie_pretrain_rt_gd.json \
#    --use_rt_task \
#    --vlnbert cmt \
#    --use_clip_feat \
#    --rt_embed_dir $rt_img_dir \
#    --use_fix_rt_emb \
#    --start_from 1
