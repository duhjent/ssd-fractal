import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .fractal_block import FractalBlock


class FractalNet(nn.Module):
    def __init__(self, data_shape, n_columns, init_channels, p_ldrop, dropout_probs,
                 gdrop_ratio, gap=0, init='xavier', pad_type='zero', doubling=False,
                 consist_gdrop=True, dropout_pos='CDBR'):
        """ FractalNet
        Args:
            - data_shape: (C, H, W, n_classes). e.g. (3, 32, 32, 10) - CIFAR 10.
            - n_columns: the number of columns
            - init_channels: the number of out channels in the first block
            - p_ldrop: local drop prob
            - dropout_probs: dropout probs (list)
            - gdrop_ratio: global droppath ratio
            - gap: pooling type for last block
            - init: initializer type
            - pad_type: padding type of conv
            - doubling: if True, doubling by 1x1 conv in front of the block.
            - consist_gdrop
            - dropout_pos: the position of dropout
                - CDBR (default): conv-dropout-BN-relu
                - CBRD: conv-BN-relu-dropout
                - FD: fractal_block-dropout
        """
        super().__init__()
        assert dropout_pos in ['CDBR', 'CBRD', 'FD']

        self.B = len(dropout_probs) # the number of blocks
        self.consist_gdrop = consist_gdrop
        self.gdrop_ratio = gdrop_ratio
        self.n_columns = n_columns
        C_in, H, W, n_classes = data_shape

        assert H == W
        size = H

        layers = nn.ModuleList()
        C_out = init_channels
        total_layers = 0
        for b, p_dropout in enumerate(dropout_probs):
            print("[block {}] Channel in = {}, Channel out = {}".format(b, C_in, C_out))
            fb = FractalBlock(n_columns, C_in, C_out, p_ldrop, p_dropout,
                              pad_type=pad_type, doubling=doubling, dropout_pos=dropout_pos)
            layers.append(fb)
            if gap == 0 or b < self.B-1:
                # Originally, every pool is max-pool in the paper (No GAP).
                layers.append(nn.MaxPool2d(2))
            elif gap == 1:
                # last layer and gap == 1
                layers.append(nn.AdaptiveAvgPool2d(1)) # average pooling

            size //= 2
            total_layers += fb.max_depth
            C_in = C_out
            if b < self.B-2:
                C_out *= 2 # doubling except for last block

        print("Last featuremap size = {}".format(size))
        print("Total layers = {}".format(total_layers))
        
        classification_head = []

        if gap == 2:
            classification_head.append(nn.Conv2d(C_out, n_classes, 1, padding=0)) # 1x1 conv
            classification_head.append(nn.AdaptiveAvgPool2d(1)) # gap
            classification_head.append(nn.Flatten())
        else:
            classification_head.append(nn.Flatten())
            classification_head.append(nn.Linear(C_out * size * size, n_classes)) # fc layer

        self.layers = layers
        self.classification_head = nn.Sequential(*classification_head)

        # initialization
        if init != 'torch':
            initialize_ = {
                'xavier': nn.init.xavier_uniform_,
                'he': nn.init.kaiming_uniform_
            }[init]

            for n, p in self.named_parameters():
                if p.dim() > 1: # weights only
                    initialize_(p)
                else: # bn w/b or bias
                    if 'bn.weight' in n:
                        nn.init.ones_(p)
                    else:
                        nn.init.zeros_(p)

    def forward(self, x, deepest=False, features_only=False):
        if deepest:
            assert self.training is False
        GB = int(x.size(0) * self.gdrop_ratio) # number of samples to be dropped out via global drop-path
        out = x
        global_cols = None
        layers = self.layers[:-1] if features_only else self.layers
        for layer in layers:
            if isinstance(layer, FractalBlock):
                if not self.consist_gdrop or global_cols is None:
                    global_cols = np.random.randint(0, self.n_columns, size=[GB]) # the column to use for each global drop-path

                out = layer(out, global_cols, deepest=deepest)
            else:
                out = layer(out)

        if not features_only:
            out = self.classification_head(out)

        return out
