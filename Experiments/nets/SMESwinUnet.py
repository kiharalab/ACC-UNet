"""
Source : https://github.com/Chenyang-1024/SMESwinUnet
"""


# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)

class SMESwinUnet(nn.Module):
    def __init__(self,n_channels, n_classes=21843, zero_head=False, vis=False):
        super(SMESwinUnet, self).__init__()
        self.num_classes = n_classes
        if n_classes != 1:
            self.num_classes += 1
        self.zero_head = zero_head
        #self.config = config

        self.swin_unet = SwinTransformerSys(img_size=224,
                                patch_size=4,
                                in_chans=n_channels,
                                num_classes=self.num_classes,
                                )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1) #repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self):
        pretrained_path = 'pretrained_ckpt/swin_tiny_patch4_window7_224.pth'
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
 