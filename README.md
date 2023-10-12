<div id="top" align="center">
  
# BDEdepth
**Towards Better Data Exploitation In Self-Supervised Monocular Depth Estimation**
  [[paper link]](https://arxiv.org/abs/2309.05254)
  
  Jinfeng Liu, [Lingtong Kong](https://scholar.google.com/citations?hl=zh-CN&user=KKzKc_8AAAAJ), [Jie Yang](http://www.pami.sjtu.edu.cn/jieyang), [Wei Liu](http://www.pami.sjtu.edu.cn/weiliu)

<p align="center">
  <img src="assets/demo.gif" alt="example input output gif" width="450" />
</p>

<p align="center">
  <img src="assets/3d_vis.png" alt="example input output gif" width="700" />
</p>
BDEdepth (HRNet18 640x192 KITTI)
</div>

## Table of Contents
- [Description](#description)
- [Setup](#setup)


## Description
This is the PyTorch implementation for BDEdepth. We build it based on the DDP version of Monodepth2 (see [Monodepth2-DDP](https://github.com/Sauf4896/monodepth2-ddp)), which has several features:
* DDP training mode
* Resume from an interrupted training automatically
* Evaluate and log after each epoch
* KITTI training and evaluation
* NYUv2 training and evaluation
* Cityscapes training and evaluation
* Make3D evaluation


If you find our work useful in your research please consider citing our paper:

```
@misc{liu2023better,
      title={Towards Better Data Exploitation In Self-Supervised Monocular Depth Estimation}, 
      author={Jinfeng Liu and Lingtong Kong and Jie Yang and Wei Liu},
      year={2023},
      eprint={2309.05254},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



## Setup
Install the dependencies with:
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install scikit-image timm thop yacs opencv-python h5py joblib
```
We experiment with PyTorch 1.9.1, CUDA 11.1, Python 3.7. Other torch versions may also be okay.


## Preparing datasets
For KITTI dataset, you can prepare them as done in [Monodepth2](https://github.com/nianticlabs/monodepth2). Note that we directly train with the raw png images and do not convert them to jpgs. You also need to generate the groundtruth depth maps before training since the code will evaluate after each epoch. For the raw KITTI groundtruth (`eigen` eval split), run the following command. This will generate `gt_depths.npz` file in the folder `splits/kitti/eigen/`.
```shell
python export_gt_depth.py --data_path /home/datasets/kitti_raw_data --split eigen
```
Or if you want to use the improved KITTI groundtruth (`eigen_benchmark` eval split), please directly download it in this [link](https://www.dropbox.com/scl/fi/dg7eskv5ztgdyp4ippqoa/gt_depths.npz?rlkey=qb39aajkbhmnod71rm32136ry&dl=0). And then move the downloaded file (`gt_depths.npz`) to the folder `splits/kitti/eigen_benchmark/`.

For NYUv2 dataset, you can download the training and testing datasets as done in [StructDepth](https://github.com/SJTU-ViSYS/StructDepth).

For Make3D dataset, you can download it from [here](http://make3d.cs.cornell.edu/data.html#make3d).

For Cityscapes dataset, we follow the instructions in [ManyDepth](https://github.com/nianticlabs/manydepth). First Download `leftImg8bit_sequence_trainvaltest.zip` and `camera_trainvaltest.zip` in its [website](https://www.cityscapes-dataset.com/), and unzip them into the folder `/path/to/cityscapes`. Then preprocess CityScapes dataset using the followimg command:
```shell
python prepare_cityscapes.py \
--img_height 512 \
--img_width 1024 \
--dataset_dir /home/datasets/cityscapes \
--dump_root /home/datasets/cityscapes_preprocessed \
--seq_length 3 \
--num_threads 8
```
Remember to modify `--dataset_dir` and `--dump_root` to your own. The ground truth depth files are provided by ManyDepth in this [link](https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip), which were converted from pixel disparities using intrinsics and the known baseline. Download this and unzip into `splits/cityscapes`


## Weights
You can download model weights in this [link](https://www.dropbox.com/scl/fo/0zeefm9e4kv0fzumqp490/h?rlkey=ev09rshvarnoyj9kr1qymkppl&dl=0), including 5 checkpoint files:
* Pretrained HRNet18 on ImageNet:   `HRNet_W18_C_cosinelr_cutmix_300epoch.pth.tar` 
* Our final KITTI model (640x192) using HRNet18 backone: `kitti_hrnet18_640x192.pth`
* Our final KITTI model (640x192) using ResNet18 backone: `kitti_resnet18_640x192.pth`
* Our final Cityscapes model (512x192) using HRNet18 backbone:  `cs_hrnet18_512x192.pth`
* Our final Cityscapes model (512x192) using ResNet18 backbone:  `cs_resnet18_512x192.pth`

Note: We additionally train and evaluate on Cityscapes here. The results are listed as follows.

| model        | abs rel | sq rel | rmse  | rmse log |  a1  | a2 | a3 |
|-------------------------|-------------------|--------------------------|-----------------|------|----------------|----------------|----------------|
|  Monodepth2 (reported in ManyDepth)  | 0.129 |1.569 |6.876 | 0.187 | 0.849 |  0.957 | 0.983 |
|  ManyDepth  |    0.114 |  1.193 |  6.223 |  0.170  |  **0.875** |  0.967 |  0.989  |
|  Ours(ResNet18) | 0.116 |    1.107 |    6.061 |    0.168 |    0.868 |    0.965 |    0.989 |
|  Ours(HRNet18) |  **0.112**  |    **1.027** |    **5.862** |    **0.163** |    0.874 |    **0.968** |    **0.990** |


## Training
Before training, move the pretrained HRNet18 weights, `HRNet_W18_C_cosinelr_cutmix_300epoch.pth.tar`, to the folder `BDEdepth/hrnet_IN_pretrained`.

```shell
cd /path/to/BDEdepth
mkdir hrnet_IN_pretrained
mv /path/to/HRNet_W18_C_cosinelr_cutmix_300epoch.pth.tar ./hrnet_IN_pretrained
```

And you can see the training scripts in [run_kitti.sh](./run_kitti.sh), [run_nyu.sh](./run_nyu.sh) and [run_cityscapes.sh](./run_cityscapes.sh). Take the KITTI script as an example:
```shell
# CUDA_VISIBLE_DEVICES=0 python train.py \

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
--data_path /home/datasets/kitti_raw_data \
--dataset kitti \
--log_dir /home/jinfengliu/logs \
--exp_name mono_kitti \
--backbone hrnet \
--num_layers 18 \
--width 640 \
--height 192 \
--num_scales 1 \
--batch_size 10 \
--lr_sche_type step \
--learning_rate 1e-4 \
--eta_min 5e-6 \
--num_epochs 20 \
--decay_step 15 \
--decay_rate 0.1 \
--log_frequency 400 \
--save_frequency 400 \
--resume \
--use_local_crop \
--use_patch_reshuffle \
# --pretrained_path xxxx/ckpt.pth
```
Use `CUDA_VISIBLE_DEVICES=0 python train.py` to train with a single GPU. If you want to train with two or more GPUs, then use `CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py` for DDP training.

Use `--data_path` flag to specify the dataset folder.

Use `--log_dir` flag to specify the logging folder.

Use `--exp_name` flag to specify the experiment name.

All output files (checkpoints, logs and tensorboard) will be saved in the directory `{log_dir}/{exp_name}`.

Use `--backbone` flag to choose the depth encoder backbone, resnet or hrnet.

Use `--use_local_crop` flag to enable the resizing-cropping augmentation.

Use `--use_patch_reshuffle` flag to enable the splitting-permuting augmentation.

Use `--pretrained_path` flag to load a pretrained checkpoint if necessary.

Use `--split` flag to specify the training split on KITTI (see [Monodepth2](https://github.com/nianticlabs/monodepth2)), and default is eigen_zhou.

Look at [options.py](./options.py) to see the range of other training options.



## Evaluation
You can see the evaluation script in [evaluate.sh](./evaluate.sh). 

```shell
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py \
--pretrained_path ./kitti_hrnet18_512x192.pth \
--backbone hrnet \
--num_layers 18 \
--batch_size 12 \
--width 640 \
--height 192 \
--kitti_path /home/datasets/kitti_raw_data \
--make3d_path /home/datasets/make3d \
--cityscapes_path /home/datasets/cityscapes \
--nyuv2_path /home/datasets/nyu_v2 
# --post_process
```
This script will evaluate on KITTI (both raw and improved GT), NYUv2, Make3D and Cityscapes together. If you don't want to evaluate on some of these datasets, for example KITTI, just do not specify the corresponding `--kitti_path` flag. It will only evaluate on the datasets which you have specified a path flag.

If you want to evalute with post-processing, add the `--post_process` flag.


## Prediction

### Prediction for a single image
You can predict scaled disparity for a single image with:

```shell
python test_simple.py --image_path folder/test_image.jpg --pretrained_path ./final_hrnet18_640x192.pth --backbone hrnet --height 192 --width 640 --save_npy
```

The `--image_path` flag can also be a directory containing several images. In this setting, the script will predict all the images (use `--ext` to specify png or jpg) in the directory:

```shell
python test_simple.py --image_path folder --pretrained_path ./final_hrnet18_640x192.pth --backbone hrnet --height 192 --width 640 --ext png --save_npy
```

### Prediction for a video

```shell
python test_video.py --image_path folder --pretrained_path ./final_hrnet18_640x192.pth --backbone hrnet --height 192 --width 640 --ext png
```
Here the `--image_path` flag should be a directory containing several video frames. Note that these video frame files should be named in an ascending numerical order. For example, the first frame is named as `0000.png`, the second frame is named as `0001.png`, and etc. Then the script will output a GIF file.

## Acknowledgement
We have used codes from other wonderful open-source projects,
[SfMLearner](https://github.com/tinghuiz/SfMLearner/tree/master),
[Monodepth2](https://github.com/nianticlabs/monodepth2), [ManyDepth](https://github.com/nianticlabs/manydepth),[StructDepth](https://github.com/SJTU-ViSYS/StructDepth), [PlaneDepth](https://github.com/svip-lab/PlaneDepth) and [RA-Depth](https://github.com/hmhemu/RA-Depth). Thanks for their excellent works!
