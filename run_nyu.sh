## follow the settings in StructDepth

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
CUDA_VISIBLE_DEVICES=0 python train.py \
--data_path /home/datasets/nyu_v2 \
--dataset nyuv2 \
--log_dir /home/jinfengliu/logs \
--exp_name mono_nyu \
--backbone hrnet \
--num_layers 18 \
--width 384 \
--height 288 \
--max_depth 10 \
--num_scales 1 \
--batch_size 16 \
--lr_sche_type step \
--learning_rate 1e-4 \
--eta_min 5e-6 \
--num_epochs 50 \
--decay_step 26 36 \
--decay_rate 0.1 \
--log_frequency 400 \
--save_frequency 400 \
--resume \
--use_local_crop 
# --use_patch_reshuffle 
# --pretrained_path xxxx/ckpt.pth
