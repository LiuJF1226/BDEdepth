# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 resize_ratio=[1.2, 2.0],
                 local_crop=False,
                 split_ratio=[0.1, 0.9],
                 patch_reshuffle=False,
                 is_train=False,
                 img_ext='.png'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.resize_ratio_lower = resize_ratio[0]
        self.resize_ratio_upper = resize_ratio[1]
        self.local_crop = local_crop

        self.split_ratio_lower = split_ratio[0]
        self.split_ratio_upper = split_ratio[1]
        self.patch_reshuffle = patch_reshuffle

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        # self.load_depth = self.check_depth()
        self.load_depth = False

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """

        if self.local_crop:
            resize_ratio = (self.resize_ratio_upper - self.resize_ratio_lower) * random.random() + self.resize_ratio_lower
            height_re = int(self.height * resize_ratio)
            width_re = int(self.width * resize_ratio)        
            w0 = int((width_re - self.width) * random.random())
            h0 = int((height_re - self.height) * random.random())
            self.resize_local = transforms.Resize((height_re, width_re), interpolation=self.interp)
            box = (w0, h0, w0+self.width, h0+self.height)
            gridx, gridy = np.meshgrid(np.linspace(-1, 1, width_re), np.linspace(-1, 1, height_re))
            gridx = torch.from_numpy(gridx)
            gridy = torch.from_numpy(gridy)
            grid = torch.stack([gridx, gridy], dim=0)
            inputs[("grid_local")] = grid[:, h0 : h0+self.height, w0 : w0+self.width].clone()  
            inputs[("ratio_local")] = torch.tensor([resize_ratio])


        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
                    if self.local_crop:
                        if i == 0:
                            inputs[(n + "_local", im, i)] = self.resize_local(inputs[(n, im, -1)]).crop(box)
                        else:
                            inputs[(n + "_local", im, i)] = self.resize[i](inputs[(n + "_local", im, i - 1)])
                if self.patch_reshuffle:
                    if im == 0:
                        ### Split-Permute as depicted in paper (vertical + horizontal)
                        img = inputs[(n, im, 0)]
                        newimg = img.copy()
                        ratio_x = random.random() * (self.split_ratio_upper - self.split_ratio_lower) + self.split_ratio_lower
                        ratio_y = random.random() * (self.split_ratio_upper - self.split_ratio_lower) + self.split_ratio_lower
                        w_i = int(self.width * ratio_x)
                        patch1 = img.crop((0, 0, w_i, self.height)).copy()
                        patch2 = img.crop((w_i, 0, self.width, self.height)).copy()
                        newimg.paste(patch2, (0, 0))
                        newimg.paste(patch1, (self.width-w_i, 0))     
                        h_i = int(self.height * ratio_y)
                        patch1 = newimg.crop((0, 0, self.width, h_i)).copy()
                        patch2 = newimg.crop((0, h_i, self.width, self.height)).copy()
                        newimg.paste(patch2, (0, 0))
                        newimg.paste(patch1, (0, self.height-h_i))      
                        inputs[(n + "_reshuffle", im, 0)] = newimg
                        inputs[("split_xy")] = torch.tensor([self.width-w_i, self.height-h_i])         

                        ### Split-Permute (vertical or horizontal, randomly choose one)
                        # img = inputs[(n, im, 0)]
                        # newimg = img.copy()
                        # ratio = random.random() * (self.split_ratio_upper - self.split_ratio_lower) + self.split_ratio_lower
                        # if random.random() > 0.5:
                        #     w_i = int(self.width * ratio)
                        #     patch1 = img.crop((0, 0, w_i, self.height)).copy()
                        #     patch2 = img.crop((w_i, 0, self.width, self.height)).copy()
                        #     newimg.paste(patch2, (0, 0))
                        #     newimg.paste(patch1, (self.width-w_i, 0))
                        #     inputs[(n + "_reshuffle", im, 0)] = newimg
                        #     inputs[("split_xy")] = torch.tensor([self.width-w_i, 0])
                        # else:
                        #     h_i = int(self.height * ratio)
                        #     patch1 = img.crop((0, 0, self.width, h_i)).copy()
                        #     patch2 = img.crop((0, h_i, self.width, self.height)).copy()
                        #     newimg.paste(patch2, (0, 0))
                        #     newimg.paste(patch1, (0, self.height-h_i))                           
                        #     inputs[(n + "_reshuffle", im, 0)] = newimg
                        #     inputs[("split_xy")] = torch.tensor([0, self.height-h_i])          

        for k in list(inputs):
            f = inputs[k]
            if "color" in k[0]:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
