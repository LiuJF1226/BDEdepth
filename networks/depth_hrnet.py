

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from collections import OrderedDict
from layers import *
from timm.models.layers import trunc_normal_
from .hrnet import hrnet18

class DepthEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained):
        super(DepthEncoder, self).__init__()
        assert num_layers==18

        if num_layers==18:
            self.encoder = hrnet18(pretrained)

        self.num_ch_enc = self.encoder.num_ch_enc

        self.context_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.context_bn = nn.BatchNorm2d(64)
        self.context_conv.weight = nn.Parameter(self.encoder.conv1.weight.clone())
        self.context_bn.weight = nn.Parameter(self.encoder.bn1.weight.clone())
        self.context_bn.bias = nn.Parameter(self.encoder.bn1.bias.clone())    


    def forward(self, x):
        x = (x - 0.45) / 0.225
        self.features = self.encoder(x)
        context = self.encoder.relu(self.context_bn(self.context_conv(x)))
        self.features = [context] + self.features
        return self.features


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), columns=3, num_output_channels=1):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales
        self.num_ch_enc = list(num_ch_enc)
        self.num_ch_enc = [self.num_ch_enc[0]] + self.num_ch_enc
        self.grid_ch = [ch//2 for ch in self.num_ch_enc]
        self.grid_ch[0] = 16
        # self.grid_ch = [16,32,32, 64, 128, 256]
        self.columns = columns

        self.convs = OrderedDict()
        for i in range(len(self.grid_ch)):
            for j in range(self.columns):
                if j == 0:
                    self.convs[('lateral', i, j)] = ResidualBlock(self.num_ch_enc[i], self.grid_ch[i], stride=1)
                else:
                    self.convs[('lateral', i, j)] = ResidualBlock(self.grid_ch[i], self.grid_ch[i], stride=1)

                if i < len(self.grid_ch)-1:
                    self.convs[('upconv', i, j)] = UpsampleBlock(self.grid_ch[i+1], self.grid_ch[i], stride=1)

            if i in self.scales:
                self.convs[("dispconv", i)] = Conv3x3(self.grid_ch[i], self.num_output_channels)
                

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        self.outputs = {}
        lateral_tmp = [0 for _ in range(self.columns)]
      
        for i in range(len(self.grid_ch)-1, -1, -1):
            x = input_features[i]
            for j in range(self.columns):
                x = self.convs[('lateral', i, j)](x)
                if i < len(self.grid_ch)-1:
                    x = x + lateral_tmp[j]
                if i > 0:
                    lateral_tmp[j] = self.convs[('upconv', i-1, j)](x)

            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
  
        return self.outputs

