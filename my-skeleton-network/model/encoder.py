import torch.nn as nn
import torch
import numpy as np
from .skeleton_operators import SkeletonConvolution, SkeletonPool, SkeletonUnPool


class EncBasicBlock(nn.Module):
    # basic block for encoder including the convolution and pooling with ReLu and batch normalization
    def __init__(self, args, in_channel, out_channel, topology, ee_id, layer_idx, dynamic=True):
        super().__init__()
        joint_num = len(topology)

        kernel_size = (joint_num, args.dynamic_kernel_size) if dynamic else (joint_num, args.static_kernel_size)
        hidden_channel = out_channel // 2

        self.conv1by1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        self.conv1 = SkeletonConvolution(in_channels=in_channel,
                                         out_channels=hidden_channel,
                                         k_size=kernel_size, stride=1,
                                         pad_size=(0, kernel_size[1] // 2),
                                         topology=topology,
                                         neighbor_dist=args.neighbor_dist_thresh,
                                         ee_id=ee_id)
        self.bn1 = nn.BatchNorm2d(num_features=hidden_channel)
        self.lkrelu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = SkeletonConvolution(in_channels=hidden_channel,
                                         out_channels=out_channel,
                                         k_size=kernel_size,
                                         stride=1,
                                         pad_size=(0, kernel_size[1] // 2),
                                         topology=topology,
                                         neighbor_dist=args.neighbor_dist_thresh,
                                         ee_id=ee_id)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.lkrelu2 = nn.LeakyReLU(inplace=True)

        self.pool = SkeletonPool(topology=topology, ee_id=ee_id, layer_idx=layer_idx)
        self.new_topology, self.new_ee_id, self.expand_num = self.pool.new_topology, self.pool.new_ee_id, self.pool.merge_nums

    def forward(self, x):
        identity = self.conv1by1(x)
        out = self.conv1(x)
        out = self.lkrelu1(self.bn1(out))
        out = self.conv2(out)
        out = self.lkrelu2(self.bn2(out) + identity)
        out = self.pool(out)
        return out


class Encoder(nn.Module):
    def __init__(self, args, init_topology, init_ee_id):
        super().__init__()
        self.topologies = [init_topology]
        self.ee_id_list = [init_ee_id]
        self.expand_num_list = []
        self.in_channels, self.out_channels = [7, 64], [32, 128]

        self.enc_layers = nn.ModuleList([EncBasicBlock(args=args, in_channel=self.in_channels[i],
                                                        out_channel=self.out_channels[i], topology=self.topologies[i],
                                                        ee_id=self.ee_id_list[i], layer_idx=i+1)
                                          for i in range(len(self.in_channels))])
        
        # self.to(args.cuda_device)
        self.apply(init_weights)

    def forward(self, x, s_latent):
        # [B, 7, joint_num, frame] -> [B, 32, joint_num, frame]
        # [B, 32, joint_num, frame] -> [B, 64, joint_num, frame]
        # [B, 64, joint_num, frame] -> [B, 128, joint_num, frame]
        out = x
        for layer in self.enc_layers:
            out = layer(out)
            if layer is self.enc_layers[0]:
                out = torch.cat([out, s_latent], dim=1)
        return out

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.01)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class StaticEncoder(nn.Module):
    """
    Encode the static offset with shape [B, 3, J, frame_num] into
    latent static tensor with shape [B, 32, J, frame_num]
    """
    def __init__(self, args, init_topology, init_ee_id):
        super().__init__()

        self.in_channel = 3
        self.out_channel = 32
        self.enc_layer = EncBasicBlock(args=args,
                                       in_channel=self.in_channel,
                                       out_channel=self.out_channel,
                                       topology=init_topology,
                                       ee_id=init_ee_id,
                                       layer_idx=1,
                                       dynamic=False)
        self._init_weights()
        # self.to(args.cuda_device)

    def forward(self, x):
        out = self.enc_layer(x)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
