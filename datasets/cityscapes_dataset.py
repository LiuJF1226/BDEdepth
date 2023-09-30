# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
import os
import json
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class CityscapesDataset(MonoDataset):
    """Cityscapes dataset - this expects triplets of images concatenated into a single wide image,
    which have had the ego car removed (bottom 25% of the image cropped)
    """

    def __init__(self, *args, **kwargs):
        super(CityscapesDataset, self).__init__(*args, **kwargs)

        if self.is_train:
            self.RAW_WIDTH = 1024
            self.RAW_HEIGHT = 384
        else:
            self.RAW_WIDTH = 2048
            self.RAW_HEIGHT = 1024       

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits

        txt file is of format:
            ulm ulm_000064_000012
        """
        city, frame_name = self.filenames[index].split()
        side = None
        return city, frame_name, side

    def check_depth(self):
        return False

    def load_intrinsics(self, city, frame_name):
        # adapted from sfmlearner
        if self.is_train:
            camera_file = os.path.join(self.data_path, city, "{}_cam.txt".format(frame_name))
            camera = np.loadtxt(camera_file, delimiter=",")
            fx = camera[0]
            fy = camera[4]
            u0 = camera[2]
            v0 = camera[5]
            intrinsics = np.array([[fx, 0, u0, 0],
                                [0, fy, v0, 0],
                                [0,  0,  1, 0],
                                [0,  0,  0, 1]]).astype(np.float32)

            intrinsics[0, :] /= self.RAW_WIDTH
            intrinsics[1, :] /= self.RAW_HEIGHT
        else:
            split = "test"  # if self.is_train else "val"
            camera_file = os.path.join(self.data_path, 'camera',
                                    split, city, frame_name + '_camera.json')
            with open(camera_file, 'r') as f:
                camera = json.load(f)
            fx = camera['intrinsic']['fx']
            fy = camera['intrinsic']['fy']
            u0 = camera['intrinsic']['u0']
            v0 = camera['intrinsic']['v0']
            intrinsics = np.array([[fx, 0, u0, 0],
                                [0, fy, v0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]]).astype(np.float32)
            intrinsics[0, :] /= self.RAW_WIDTH
            intrinsics[1, :] /= self.RAW_HEIGHT * 0.75            
        return intrinsics

    def get_color(self, city, frame_name, side, do_flip):
        if side is not None:
            raise ValueError("Cityscapes dataset doesn't know how to deal with sides")

        inputs = {}
        if self.is_train:
            color = self.loader(self.get_image_path(city, frame_name))
            color = np.array(color)

            h = color.shape[0] // 3
            inputs[("color", -1, -1)] = pil.fromarray(color[:h, :])
            inputs[("color", 0, -1)] = pil.fromarray(color[h:2*h, :])
            inputs[("color", 1, -1)] = pil.fromarray(color[2*h:, :])

            if do_flip:
                for key in inputs:
                    inputs[key] = inputs[key].transpose(pil.FLIP_LEFT_RIGHT)
        else:
            color = self.loader(self.get_image_path(city, frame_name))
            w, h = color.size
            crop_h = h * 3 // 4
            color = color.crop((0, 0, w, crop_h))
            inputs[("color", 0, -1)] = color
        return inputs

    def get_image_path(self, city, frame_name):
        if self.is_train:
            return os.path.join(self.data_path, city, "{}.png".format(frame_name))
        else:
            folder = "leftImg8bit_sequence"
            split = "test"
            image_path = os.path.join(
                self.data_path, folder, split, city, frame_name + '_leftImg8bit.png')
            return image_path            
