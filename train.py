# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time
import logging

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from options import opts
import torch.distributed as dist
import os
import json
from utils import *
from kitti_utils import *
from layers import *


import datasets
from networks import *
# from IPython import embed


class Trainer:
    def __init__(self, options):
        self.opt = options

        self.log_path = os.path.join(self.opt.log_dir, self.opt.exp_name)
        if self.opt.global_rank == 0:
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            self.save_opts()
            if not os.path.exists(os.path.join(self.log_path, 'ckpt.pth')):
                setup_logging(os.path.join(self.log_path, 'logger.log'), rank=self.opt.global_rank)
                logging.info("Experiment is named: %s", self.opt.exp_name)
                logging.info("Saving to: %s", os.path.abspath(self.log_path))
                logging.info("GPU numbers: %d", self.opt.world_size)
            else:
                setup_logging(os.path.join(self.log_path, 'logger.log'), filemode='a', rank=self.opt.global_rank)

            self.writers = {}
            for mode in ["train"]:
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, "tensorboard", mode))

        if self.opt.world_size > 1:
            dist.barrier()

        self.device = torch.device('cuda', self.opt.local_rank)

        if self.opt.seed > 0:
            self.set_seed(self.opt.seed)
        else:
            cudnn.benchmark = True

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        self.ep_start = 0
        self.batch_start = 0
        self.step = 0

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")


        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "nyuv2": datasets.NYUDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        if self.opt.dataset == "kitti":
            fpath = os.path.join(os.path.dirname(__file__), "splits/kitti", self.opt.split, "{}_files.txt")
            fpath_test = os.path.join(os.path.dirname(__file__), "splits/kitti", self.opt.eval_split, "{}_files.txt")
        elif self.opt.dataset == "kitti_odom":
            pass
        elif self.opt.dataset == "nyuv2":
            fpath = os.path.join(os.path.dirname(__file__), "splits/nyuv2", "{}_files.txt")
            fpath_test = os.path.join(os.path.dirname(__file__), "splits/nyuv2", "{}_files.txt")
        else:
            pass


        train_filenames = readlines(fpath.format("train"))
        test_filenames = readlines(fpath_test.format("test"))
        img_ext = '.jpg' if self.opt.jpg else '.png'

        num_train_samples = len(train_filenames)
        self.num_steps_per_epoch = num_train_samples // self.opt.world_size // self.opt.batch_size
        self.num_total_steps = self.num_steps_per_epoch * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width, self.opt.frame_ids, self.opt.num_scales, local_crop=self.opt.use_local_crop, patch_reshuffle=self.opt.use_patch_reshuffle, is_train=True, img_ext=img_ext)
        if self.opt.world_size > 1:
            self.sampler = datasets.CustomDistributedSampler(train_dataset, self.opt.seed)
        else:
            self.sampler = datasets.CustomSampler(train_dataset, self.opt.seed)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=False, sampler=self.sampler, num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        # for testing the model at the end of each epoch
        test_dataset = self.dataset(
            self.opt.data_path, test_filenames, self.opt.height, self.opt.width,
            [0], self.opt.num_scales, is_train=False, img_ext=img_ext)
        self.test_loader = DataLoader(
            test_dataset, self.opt.batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)
        if self.opt.dataset == "kitti":
            gt_path = os.path.join(os.path.dirname(__file__), "splits/kitti", self.opt.eval_split, "gt_depths.npz")
            self.gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
        else:
            pass

        # create models
        if self.opt.backbone == "resnet":
            self.models["encoder"] = depth_resnet.DepthEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["depth"] = depth_resnet.DepthDecoder(
                self.models["encoder"].num_ch_enc, range(self.opt.num_scales))
        if self.opt.backbone == "hrnet":
            self.models["encoder"] = depth_hrnet.DepthEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["depth"] = depth_hrnet.DepthDecoder(
                self.models["encoder"].num_ch_enc, range(self.opt.num_scales))
                
        self.models["encoder"].to(self.device)
        self.models["depth"].to(self.device)


        if self.use_pose_net:
            self.models["pose_encoder"] = posenet.ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)

            self.models["pose_encoder"].to(self.device)

            self.models["pose"] = posenet.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)

            self.models["pose"].to(self.device)
 

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, range(self.opt.num_scales),
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)

        for k in self.models.keys():
            self.parameters_to_train += list(self.models[k].parameters())

        if self.opt.pretrained_path:
            if not self.opt.resume:
                self.load_pretrained_model()
            elif not os.path.exists(os.path.join(self.log_path, 'ckpt.pth')):
                self.load_pretrained_model()

        if self.opt.resume:
            checkpoint = self.load_ckpt()

        if self.opt.world_size > 1:
            for k in self.models.keys():
                self.models[k] = nn.SyncBatchNorm.convert_sync_batchnorm(
                self.models[k])
                self.models[k] = nn.parallel.DistributedDataParallel(self.models[k], device_ids=[self.opt.local_rank], output_device=self.opt.local_rank, find_unused_parameters=True)

        # optimizer settings
        if self.opt.optimizer == 'adamw':
            self.model_optimizer = torch.optim.AdamW(self.parameters_to_train,lr=self.opt.learning_rate, betas=(self.opt.beta1, self.opt.beta2),weight_decay=self.opt.weight_decay)
        elif self.opt.optimizer == 'adam':
            self.model_optimizer = torch.optim.Adam(self.parameters_to_train,lr=self.opt.learning_rate, betas=(self.opt.beta1, self.opt.beta2)) 
        elif self.opt.optimizer == 'sgd':
            self.model_optimizer = torch.optim.SGD(self.parameters_to_train,lr=self.opt.learning_rate, momentum=self.opt.momentum)
        else:
            logging.error("Optimizer '%s' not defined. Use (adamw|adam|sgd) instead", self.opt.optimizer)


        if self.opt.lr_sche_type == 'cos':
            self.model_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.model_optimizer, T_max=self.num_total_steps, eta_min=self.opt.eta_min)
        elif self.opt.lr_sche_type == 'step':
            self.model_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.model_optimizer, self.opt.decay_step, self.opt.decay_rate)

        if checkpoint:
            self.model_optimizer.load_state_dict(checkpoint["optimizer"])
            self.model_lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            del checkpoint

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in range(self.opt.num_scales):
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        logging.info("Using split: %s", self.opt.split)
        logging.info("There are {:d} training items and {:d} test items\n".format(len(train_dataset), len(test_dataset)))

        if self.opt.world_size > 1:
            dist.barrier()


    def set_seed(self, seed=1234):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        for self.epoch in range(self.ep_start, self.opt.num_epochs):
            self.run_epoch()
            if self.opt.lr_sche_type == "step":
                self.model_lr_scheduler.step()
            with torch.no_grad():
                if self.opt.dataset == "kitti":
                    self.test_kitti()
                elif self.opt.dataset == "nyuv2":
                    self.test_nyuv2()
                else:
                    pass
            if self.opt.global_rank == 0:
                self.save_model(ep_end=True)

    def test_nyuv2(self):
        logging.info(" ")
        logging.info("Test the model at epoch {} \n".format(self.epoch))

        self.set_eval()
        pred_depths = []
        gt_depths = []

        for idx, (color, depth) in enumerate(self.test_loader):
            if self.opt.global_rank == 0:
                print("{}/{}".format(idx+1, len(self.test_loader)), end='\r')
            input_color = color.to(self.device)

            output = self.models["depth"](self.models["encoder"](input_color))
            pred_disp, _ = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0]
            
            gt_depth = depth
            _, h, w = gt_depth.shape
            pred_depth = 1 / pred_disp
            pred_depth = F.interpolate(pred_depth.unsqueeze(0), (h, w), mode="nearest")[0]
            pred_depths.append(pred_depth)
            gt_depths.append(gt_depth)
        pred_depths = torch.cat(pred_depths, dim=0)
        gt_depths = torch.cat(gt_depths, dim=0)

        errors = []
        ratios = []
        for i in range(pred_depths.shape[0]):    
            pred_depth = pred_depths[i]
            gt_depth = gt_depths[i]
            mask = (gt_depth > 0) & (gt_depth < 10)
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            ratio = torch.median(gt_depth) / torch.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio         
            pred_depth[pred_depth > 10] = 10
            errors.append(compute_depth_errors(gt_depth, pred_depth))

        ratios = np.array(ratios)
        med = np.median(ratios)
        std = np.std(ratios / med)

        logging.info(" Mono evaluation - using median scaling")
        logging.info(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, std))

        mean_errors = np.array(errors).mean(0)

        logging.info(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        logging.info(("{: 8.3f} | " * 7 + "\n").format(*mean_errors.tolist()))
        self.set_train()

    def test_kitti(self):
        """Test the model on a single minibatch
        """
        logging.info(" ")
        logging.info("Test the model at epoch {} \n".format(self.epoch))

        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
        # Models which were trained with stereo supervision were trained with a nominal baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore, to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
        STEREO_SCALE_FACTOR = 5.4
        self.set_eval()

        pred_disps = []
        for idx, data in enumerate(self.test_loader):
            if self.opt.global_rank == 0:
                print("{}/{}".format(idx+1, len(self.test_loader)), end='\r')
            input_color = data[("color", 0, 0)].to(self.device)
            output = self.models["depth"](self.models["encoder"](input_color))
            pred_disp, _ = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            pred_disps.append(pred_disp.cpu()[:, 0])
        pred_disps = torch.cat(pred_disps, dim=0)

        errors = []
        ratios = []
        for i in range(pred_disps.shape[0]):
            gt_depth = torch.from_numpy(self.gt_depths[i])
            gt_height, gt_width = gt_depth.shape[:2]
            pred_disp = pred_disps[i:i+1].unsqueeze(0)
            pred_disp = F.interpolate(pred_disp, (gt_height, gt_width), mode="bilinear", align_corners=False)
            pred_depth = 1 / pred_disp[0, 0, :]
            if self.opt.eval_split == "eigen":
                mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH)
                crop_mask = torch.zeros_like(mask)
                crop_mask[
                        int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                        int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
                mask = mask * crop_mask
            else:
                mask = gt_depth > 0

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]
            if self.opt.use_stereo and len(self.opt.frame_ids)==1:
                pred_depth *= STEREO_SCALE_FACTOR
            else:
                ratio = torch.median(gt_depth) / torch.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio  
            pred_depth = torch.clamp(pred_depth, MIN_DEPTH, MAX_DEPTH)
            errors.append(compute_depth_errors(gt_depth, pred_depth))

        if self.opt.use_stereo and len(self.opt.frame_ids)==1:
            logging.info(" Stereo evaluation - disabling median scaling")
            logging.info(" Scaling by {}".format(STEREO_SCALE_FACTOR))
        else:
            ratios = np.array(ratios)
            med = np.median(ratios)
            std = np.std(ratios / med)
            logging.info(" Mono evaluation - using median scaling")
            logging.info(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, std))

        mean_errors = np.array(errors).mean(0)

        logging.info(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        logging.info(("{: 8.3f} | " * 7 + "\n").format(*mean_errors.tolist()))
        self.set_train()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        logging.info("Training epoch {}\n".format(self.epoch))

        self.sampler.set_epoch(self.epoch)
        self.sampler.set_start_iter(self.batch_start*self.opt.batch_size)
        self.set_train()

        if self.opt.world_size > 1:
            dist.barrier()

        start_data_time = time.time()
        for batch_idx, inputs in enumerate(self.train_loader):
            self.step += 1
            start_fp_time = time.time()
            outputs, losses = self.process_batch(inputs)

            start_bp_time = time.time()
            self.model_optimizer.zero_grad()

            losses["loss"].backward()
            if self.opt.clip_grad != -1:
                for params in self.model_optimizer.param_groups:
                    params = params['params']
                    nn.utils.clip_grad_norm_(params, max_norm=self.opt.clip_grad)

            self.model_optimizer.step()

            if self.opt.lr_sche_type == "cos":
                self.model_lr_scheduler.step()

            # compute the process time
            data_time = start_fp_time - start_data_time
            fp_time = start_bp_time - start_fp_time
            bp_time = time.time() - start_bp_time

            # logging
            if ((batch_idx+self.batch_start) % self.opt.log_frequency == 0):
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                if self.opt.world_size > 1:
                    dist.barrier()
                    for k in losses.keys():
                        dist.all_reduce(losses[k], op=dist.ReduceOp.SUM)
                        losses[k] /= self.opt.world_size
                    dist.barrier()
                if self.opt.global_rank == 0:
                    self.log_time(batch_idx+self.batch_start, data_time, fp_time, bp_time, losses)
                    self.log_tensorboard("train", inputs, outputs, losses)

            # save ckpt
            if ((batch_idx+self.batch_start)>0 and (batch_idx+self.batch_start) % self.opt.save_frequency == 0):
                if self.opt.global_rank == 0:
                    self.save_model(batch_idx=batch_idx+self.batch_start+1)
            if self.opt.world_size > 1:
                dist.barrier()
            start_data_time = time.time()

        self.batch_start = 0

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            try:
                inputs[key] = ipt.to(self.device)
            except:
                pass

        features = self.models["encoder"](inputs[("color_aug", 0, 0)])
        outputs = self.models["depth"](features)

        if self.opt.use_local_crop:
            x = self.models["encoder"](inputs[("color_local_aug", 0, 0)])
            o = self.models["depth"](x)
            for i in range(self.opt.num_scales):
                outputs[("disp_local", i)] = o[("disp", i)]
        if self.opt.use_patch_reshuffle:
            x = self.models["encoder"](inputs[("color_reshuffle_aug", 0, 0)])
            o = self.models["depth"](x)
            for i in range(self.opt.num_scales):
                outputs[("disp_reshuffle", i)] = o[("disp", i)]

                # restore the reshuffled disparity maps
                all_disp = []
                for b in range(self.opt.batch_size):
                    ### Split-Permute as depicted in paper (vertical + horizontal)
                    split_x = inputs[("split_xy")][b][0].item()
                    split_y = inputs[("split_xy")][b][1].item()
                    split_x = round(split_x / (2 ** i))
                    split_y = round(split_y / (2 ** i))
                    disp_reshuffle = outputs[("disp_reshuffle", i)][b]   #1*H*W
                    patch1 = disp_reshuffle[:, 0:split_y, :]
                    patch2 = disp_reshuffle[:, split_y:, :]
                    disp_restore = torch.cat([patch2, patch1], dim=1)
                    patch1 = disp_restore[:, :, 0:split_x]
                    patch2 = disp_restore[:, :, split_x:]
                    disp_restore = torch.cat([patch2, patch1], dim=2)
                    all_disp.append(disp_restore)

                    ### Split-Permute (vertical or horizontal, randomly choose one)
                    # split_x = inputs[("split_xy", i)][b][0].item()
                    # split_y = inputs[("split_xy", i)][b][1].item()
                    # split_x = round(split_x / (2 ** i))
                    # split_y = round(split_y / (2 ** i))
                    # disp_reshuffle = outputs[("disp_reshuffle", i)][b]   #1*H*W
                    # if split_x == 0:
                    #     patch1 = disp_reshuffle[:, 0:split_y, :]
                    #     patch2 = disp_reshuffle[:, split_y:, :]
                    #     disp_restore = torch.cat([patch2, patch1], dim=1)
                    # else:
                    #     patch1 = disp_reshuffle[:, :, 0:split_x]
                    #     patch2 = disp_reshuffle[:, :, split_x:]
                    #     disp_restore = torch.cat([patch2, patch1], dim=2)
                    # all_disp.append(disp_restore)
                disp_restore = torch.stack(all_disp, dim=0)
                outputs[("disp_reshuffle", i)] = disp_restore

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            pose_inputs = torch.cat(
                [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

            pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

      
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in range(self.opt.num_scales):
            # original
            disp = outputs[("disp", scale)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
                outputs[("sample", frame_id, scale)] = pix_coords
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

            # patch_reshuffled
            if self.opt.use_patch_reshuffle:
                disp = outputs[("disp_reshuffle", scale)]
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                for _, frame_id in enumerate(self.opt.frame_ids[1:]):
                    if frame_id == "s":
                        T = inputs["stereo_T"]
                    else:
                        T = outputs[("cam_T_cam", 0, frame_id)]
                    cam_points = self.backproject_depth[source_scale](
                        depth, inputs[("inv_K", source_scale)])
                    pix_coords = self.project_3d[source_scale](
                        cam_points, inputs[("K", source_scale)], T)
                    outputs[("color_reshuffle", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)].clone(),
                        pix_coords,
                        padding_mode="border", align_corners=True)  

            # local crop
            if self.opt.use_local_crop:
                disp = outputs[("disp_local", scale)]
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                for _, frame_id in enumerate(self.opt.frame_ids[1:]):
                    if frame_id == "s":
                        T = inputs["stereo_T"]
                    else: 
                        T = outputs[("cam_T_cam", 0, frame_id)]
                    Rt_Rc = torch.zeros_like(T).to(self.device)
                    gx0 = (inputs[("grid_local")][:, 0, 0, -1] + inputs[("grid_local")][:, 0, 0, 0]) / 2.
                    gy0 = (inputs[("grid_local")][:, 1, -1, 0] + inputs[("grid_local")][:, 1, 0, 0]) / 2.
                    f = (inputs[("grid_local")][:, 0, 0, -1] - inputs[("grid_local")][:, 0, 0, 0]) / 2.
                    fx = inputs[("K", 0)][0, 0, 0] / self.opt.width
                    fy = inputs[("K", 0)][0, 1, 1] / self.opt.height
                    Rc_v = torch.stack([-gx0/(2*fx), -gy0/(2*fy), f], dim=1)
                    Rc = torch.eye(3).to(self.device)
                    Rc = Rc[None, :, :].repeat(Rc_v.shape[0], 1, 1)
                    Rc[:, :, 2] = Rc_v
                    # outputs[("Rc", f_i)] = Rc
                    Rt_Rc[:, :3, :3] = torch.matmul(Rc, torch.matmul(T[:, :3, :3], torch.inverse(Rc)))
                    Rt_Rc[:, :3, 3:4] = torch.matmul(Rc, T[:, :3, 3:4])                    
                    T = Rt_Rc
                    cam_points = self.backproject_depth[source_scale](
                        depth, inputs[("inv_K", source_scale)])
                    pix_coords = self.project_3d[source_scale](
                        cam_points, inputs[("K", source_scale)], T)
                    outputs[("color_local", frame_id, scale)] = F.grid_sample(
                        inputs[("color_local", frame_id, source_scale)],
                        pix_coords, padding_mode="border",align_corners=True)                        


    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss


    def compute_SI_log_depth_loss(self, pred, target, mask=None, lamda=0.5):
        # B*1*H*W  ->  B*H*W
        if mask is None:
            mask = torch.ones_like(pred).to(self.device)

        mask = mask[:, 0] 
        log_pred = torch.log(pred[:, 0]+1e-8) * mask
        log_tgt = torch.log(target[:, 0]+1e-8) * mask
     
        log_diff = log_pred - log_tgt
        valid_num = mask.sum(1).sum(1) + 1e-8
        log_diff_squre_sum = (log_diff ** 2).sum(1).sum(1)
        log_diff_sum_squre = (log_diff.sum(1).sum(1)) ** 2
        loss = log_diff_squre_sum/valid_num - lamda*log_diff_sum_squre/(valid_num**2)

        return loss.mean()


    def compute_losses_base(self, inputs, outputs, ty=""):
        loss_all = 0
        for scale in range(self.opt.num_scales):
            loss = 0
            reprojection_losses = []
            source_scale = 0

            if ty == "_local":
                disp = outputs[("disp"+ty, scale)]
                color = inputs[("color"+ty, 0, scale)]
                target = inputs[("color"+ty, 0, source_scale)]
            else:
                disp = outputs[("disp"+ty, scale)]
                color = inputs[("color", 0, scale)]
                target = inputs[("color", 0, source_scale)]                

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color"+ty, frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)
            
            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    if ty == "_local":
                        pred = inputs[("color"+ty, frame_id, source_scale)]
                    else:
                        pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection"+ty+"/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)     
            loss_all += loss
        loss_all /= self.opt.num_scales

        return loss_all


    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}

        loss_base = 0
        n = 0
        loss_base_ori = self.compute_losses_base(inputs, outputs, ty="")
        losses["loss_base_ori"] = loss_base_ori
        loss_base += loss_base_ori
        if self.opt.use_local_crop:
            loss_base_local = self.compute_losses_base(inputs, outputs, ty="_local")
            losses["loss_base_local"] = loss_base_local
            loss_base += loss_base_local
            n += 1
        if self.opt.use_patch_reshuffle:
            loss_base_reshuffle = self.compute_losses_base(inputs, outputs, ty="_reshuffle")
            losses["loss_base_reshuffle"] = loss_base_reshuffle
            loss_base += loss_base_reshuffle
            n += 1
        loss_base /= n+1
        losses["loss_base"] = loss_base


        # depth consistency loss
        loss_dc = torch.tensor(0.0).to(self.device)
        if self.opt.use_local_crop:
            loss_dc_local = 0
            for i in range(self.opt.num_scales):
                disp = outputs[("disp", i)]
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                loss_dc_i = 0
                for b in range(self.opt.batch_size):
                    disp_local = outputs[("disp_local", i)][b].clone()
                    x0 = round(self.opt.width * (inputs[("grid_local")][b, 0, 0, 0].item() - (-1)) / 2.)
                    y0 = round(self.opt.height * (inputs[("grid_local")][b, 1, 0, 0].item() - (-1)) / 2.)        
                    w = round(self.opt.width/inputs[("ratio_local")][b, 0].item())
                    h = round(self.opt.height/inputs[("ratio_local")][b, 0].item())
                    disp_local = F.interpolate(disp_local.unsqueeze(0), [h, w], mode="bilinear", align_corners=False)
                    _, depth_local = disp_to_depth(disp_local, self.opt.min_depth, self.opt.max_depth)
                    depth_local *= inputs[("ratio_local")][b, 0]
                    _, depth_from_ori = disp_to_depth(disp[b, :, y0:y0+h, x0:x0+w].clone().unsqueeze(0), self.opt.min_depth, self.opt.max_depth)
                    
                    loss_dc_i += self.compute_SI_log_depth_loss(depth_local, depth_from_ori)
                loss_dc_i /= self.opt.batch_size
                loss_dc_local += loss_dc_i
            loss_dc_local /= self.opt.num_scales
            losses["loss_dc_local"] = loss_dc_local
            loss_dc += loss_dc_local

        if self.opt.use_patch_reshuffle:
            loss_dc_reshuffle = 0
            for i in range(self.opt.num_scales):
                _, depth = disp_to_depth(outputs[("disp", i)].clone(), self.opt.min_depth, self.opt.max_depth)
                disp_restore = outputs[("disp_reshuffle", i)]
                _, depth_restore = disp_to_depth(disp_restore, self.opt.min_depth, self.opt.max_depth)
                loss_dc_reshuffle += self.compute_SI_log_depth_loss(depth_restore, depth)   
            loss_dc_reshuffle /= self.opt.num_scales
            losses["loss_dc_reshuffle"] = loss_dc_reshuffle     
            loss_dc += loss_dc_reshuffle

        losses["loss_dc"] = loss_dc
        total_loss = loss_base + 0.07 * loss_dc
        # total_loss = loss_base
        losses["loss"] = total_loss

        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())
 
    def log_time(self, batch_idx, data_time, fp_time, bp_time, losses):
        """Print a logging statement to the terminal
        """
        batch_time = data_time + fp_time + bp_time
        # time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps - self.step) * batch_time if self.step > 1 else 0
        print_string = "epoch: {:>2}/{} | batch: {:>4}/{} | data time: {:.4f}" + " | batch time: {:.3f} | loss_base: {:.4f} | loss_dc: {:.4f} | lr: {:.2e} | time left: {}"
        logging.info(print_string.format(self.epoch, self.opt.num_epochs-1,batch_idx, self.num_steps_per_epoch, data_time, batch_time, losses["loss_base"].cpu().data, losses["loss_dc"].cpu().data, self.model_optimizer.state_dict()['param_groups'][0]['lr'], sec_to_hm_str(training_time_left)))

    def log_tensorboard(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(3, self.opt.batch_size)):  # write a maxmimum of four images
            for s in range(self.opt.num_scales):
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

                if self.opt.use_local_crop:
                    for frame_id in self.opt.frame_ids:
                        writer.add_image(
                            "color_local_{}_{}/{}".format(frame_id, s, j),
                            inputs[("color_local", frame_id, s)][j].data, self.step)
                        if s == 0 and frame_id != 0:
                            writer.add_image(
                                "color_pred_local_{}_{}/{}".format(frame_id, s, j),
                                outputs[("color_local", frame_id, s)][j].data, self.step)

                    writer.add_image(
                        "disp_local_{}/{}".format(s, j),
                        normalize_image(outputs[("disp_local", s)][j]), self.step)

                    # writer.add_image(
                    #     "automask_local_{}/{}".format(s, j),
                    #     outputs["identity_selection_local/{}".format(s)][j][None, ...], self.step)

                if self.opt.use_patch_reshuffle:
                    writer.add_image(
                        "disp_reshuffle_restore_{}/{}".format(s, j),
                        normalize_image(outputs[("disp_reshuffle", s)][j]), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        to_save = self.opt.__dict__.copy()
        with open(os.path.join(self.log_path, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)
        s = os.path.split(os.path.realpath(__file__))[0]+'/'
        t = os.path.join(self.log_path, 'codes')
        os.system("rm -rf {}".format(t))
        os.system("cp -rf {} {}".format(s, t))


    def save_model(self, ep_end=False, batch_idx=0):
        """Save model weights to disk
        """

        models_dir = os.path.join(self.log_path, "ckpts_each_epoch")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = {}
        for model_name, model in self.models.items():
            if self.opt.world_size == 1:
                to_save[model_name] = model.state_dict()
            else:
                to_save[model_name] = model.module.state_dict()
        to_save['height'] = self.opt.height
        to_save['width'] = self.opt.width
        to_save['use_stereo'] = self.opt.use_stereo     
        to_save['step_in_total'] = self.step
        to_save["batch_idx"] = batch_idx
        to_save['optimizer'] = self.model_optimizer.state_dict()
        to_save['lr_scheduler'] = self.model_lr_scheduler.state_dict()

        save_ep_path = os.path.join(models_dir, "ckpt{}.pth".format(self.epoch))
        save_path = os.path.join(self.log_path, "ckpt.pth")
        if ep_end:
            to_save["epoch"] = self.epoch + 1
            torch.save(to_save, save_ep_path)
        else:   
            to_save["epoch"] = self.epoch 

        torch.save(to_save, save_path)


    def load_ckpt(self):
        """Load checkpoint to resume a training, used in training process.
        """
        logging.info(" ")
        load_path = os.path.join(self.log_path, "ckpt.pth")
        if not os.path.exists(load_path):
            logging.info("No checkpoint to resume, train from epoch 0.")
            return None

        logging.info("Resume checkpoint from {}".format(os.path.abspath(load_path)))
        checkpoint = torch.load(load_path, map_location='cpu')
        for model_name, model in self.models.items():
            model_dict = model.state_dict()
            pretrained_dict = checkpoint[model_name]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        self.ep_start = checkpoint['epoch']
        self.batch_start = checkpoint['batch_idx']
        self.step = checkpoint['step_in_total']
        logging.info("Start at eopch {}, batch index {}".format(self.ep_start, self.batch_start))
        return checkpoint

    def load_pretrained_model(self):
        """Load pretrained model(s) from disk, used for initializing.
        """
        self.opt.pretrained_path = os.path.abspath(self.opt.pretrained_path)

        assert os.path.exists(self.opt.pretrained_path), \
            "Cannot find folder {}".format(self.opt.pretrained_path)
        logging.info("Loading pretrained model from folder {}".format(self.opt.pretrained_path))

        checkpoint = torch.load(self.opt.pretrained_path, map_location='cpu')
        for model_name, model in self.models.items():
            model_dict = model.state_dict()
            pretrained_dict = checkpoint[model_name]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)



if __name__ == "__main__":
    opts.world_size = torch.cuda.device_count()
    if opts.world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(opts.local_rank)
        opts.global_rank = torch.distributed.get_rank()
    trainer = Trainer(opts)
    trainer.train()
