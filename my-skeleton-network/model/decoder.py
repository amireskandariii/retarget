import torch.nn as nn
import torch
import numpy as np
from .skeleton_operators import SkeletonConvolution, SkeletonPool, SkeletonUnPool

class DecBasicBlock(nn.Module):
    def __init__(self, args, in_channel, out_channel, topology, ee_id, expand_nums):
        super().__init__()
        
        kernel_size = (len(topology), args.dynamic_kernel_size)
        hidden_channel = out_channel // 2
        
        self.un_pool = SkeletonUnPool(un_pool_expand_nums=expand_nums)
        
        self.conv1by1 = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        
        self.conv1 = SkeletonConvolution(
            in_channels=in_channel,
            out_channels=hidden_channel,
            k_size=kernel_size,
            stride=1,
            pad_size=(0, kernel_size[1] // 2),
            topology=topology,
            neighbor_dist=args.neighbor_dist_thresh,
            ee_id=ee_id
        )
        self.bn1 = nn.BatchNorm2d(num_features=hidden_channel)
        self.lkrelu = nn.LeakyReLU(inplace=True)

        self.conv2 = SkeletonConvolution(
            in_channels=hidden_channel,
            out_channels=out_channel,
            k_size=kernel_size,
            stride=1,
            pad_size=(0, kernel_size[1] // 2),
            topology=topology,
            neighbor_dist=args.neighbor_dist_thresh,
            ee_id=ee_id
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)

    def forward(self, x, s_latent):
        out = self.un_pool(x)
        out = torch.cat([out, s_latent], dim=1)
        
        identity = self.conv1by1(out)
        
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.lkrelu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        
        return out

class Decoder(nn.Module):
    def __init__(self, args, topologies, ee_ids, expand_nums):
        super().__init__()
        in_channels = [128 + 32, 64 + 3]
        out_channels = [64, 4]
        
        self.dec_layer1 = DecBasicBlock(args, in_channels[0], out_channels[0],
                                        topologies[-1], ee_ids[-1], expand_nums[-1])
        self.dec_layer2 = DecBasicBlock(args, in_channels[1], out_channels[1],
                                        topologies[-2], ee_ids[-2], expand_nums[-2])
        self.lkrelu = nn.LeakyReLU(inplace=True)

        self._init_weights()

        # self.to(args.cuda_device)

    def forward(self, d_latent, s_latent, sx):
        out = self.dec_layer1(d_latent, s_latent)
        out = self.lkrelu(out)
        out = self.dec_layer2(out, sx)
        return out
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
