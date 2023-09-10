# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

from networks import *
from layers import disp_to_depth
from evaluate_depth import STEREO_SCALE_FACTOR


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--pretrained_path',
                        type=str,
                        help="path of model checkpoint to load")
    parser.add_argument("--backbone",
                        type=str,
                        help="backbone of depth encoder",
                        default="hrnet",
                        choices=["resnet", "hrnet"])
    parser.add_argument("--num_layers",
                        type=int,
                        help="number of resnet layers",
                        default=18,
                        choices=[18, 34, 50, 101, 152])
    parser.add_argument("--height",
                        type=int,
                        help="input image height",
                        default=192)
    parser.add_argument("--width",
                        type=int,
                        help="input image width",
                        default=640)
    parser.add_argument("--min_depth",
                        type=float,
                        help="minimum depth",
                        default=0.1)
    parser.add_argument("--max_depth",
                        type=float,
                        help="maximum depth",
                        default=100.0)
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.pretrained_path is not None

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("-> Loading model from ", args.pretrained_path)
    model = torch.load(args.pretrained_path, map_location='cpu')
    if args.backbone == "hrnet":
        depth_encoder = depth_hrnet.DepthEncoder(args.num_layers, False)
        depth_decoder = depth_hrnet.DepthDecoder(depth_encoder.num_ch_enc, scales=range(1))
    if args.backbone == "resnet":
        depth_encoder = depth_resnet.DepthEncoder(args.num_layers, False)
        depth_decoder = depth_resnet.DepthDecoder(depth_encoder.num_ch_enc, scales=range(1))
    depth_encoder.load_state_dict({k: v for k, v in model["encoder"].items() if k in depth_encoder.state_dict()})
    depth_decoder.load_state_dict({k: v for k, v in model["depth"].items() if k in depth_decoder.state_dict()})
    depth_encoder.to(device).eval()
    depth_decoder.to(device).eval()
    feed_height = args.height
    feed_width = args.width


    assert os.path.isdir(args.image_path)

    paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
    paths = sorted(paths)
 
    output_directory = args.image_path
   

    print("-> Predicting on {:d} video frames".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        ims = []
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            img_ori = np.array(input_image).copy()
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = depth_encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            
            # Saving colormapped depth image
            disp_np = disp.squeeze().cpu().numpy()
            vmax = np.percentile(disp_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)

            out = np.concatenate([img_ori, colormapped_im], axis=0)

            out = pil.fromarray(out)
            # out = out.resize((feed_width//2, feed_height), pil.LANCZOS)
            ims.append(out)
        
        # name_dest = os.path.join(output_directory, "demo.gif")
        name_dest = "demo.gif"
        ims[0].save(name_dest, save_all=True, append_images=ims[1:], duration=150, loop=0)

        print(" Save output gif to:   {}".format(name_dest))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
