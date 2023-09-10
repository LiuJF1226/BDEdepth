CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py \
--pretrained_path ./final_hrnet18_640x192.pth \
--backbone hrnet \
--num_layers 18 \
--batch_size 12 \
--width 640 \
--height 192 \
--kitti_path /home/datasets/kitti_raw_data \
--make3d_path /home/datasets/make3d \
--nyuv2_path /home/datasets/nyu_v2 \
# --post_process

