import torch.nn as nn
import torch
import numpy as np

class SkeletonConvolution(nn.Module):
    # convolution on skeleton topology
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], stride: int, 
                 padding: tuple[int, int], topology: np.ndarray, neighbor_dist: int, ee_id: list[int]):
        super(SkeletonConvolution, self).__init__()
        self.neighbor_list = find_neighbor(topology, neighbor_dist, ee_id)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result_list = [self.conv(x * torch.tensor([[[[1]]]], dtype=torch.float).to(x.device)[:, :, neighbors, :]) for neighbors in self.neighbor_list]
        return torch.cat(result_list, dim=2)
    

class SkeletonPool(nn.Module):
    # pooling on skeleton topology
    def __init__(self, topology, ee_id, layer_idx):
        super(SkeletonPool, self).__init__()
        self.old_topology = topology
        self.old_ee_id = ee_id
        # self.new_topology = pooled_topology if layer_idx == 1 else  pooled_topology2
        # self.new_ee_id = pooled_ee_id if layer_idx == 1 else pooled_ee_id2
        self.seq_list = []
        self.pooling_list = []
        self.old_joint_num = len(self.old_topology.tolist())
        self.new_joint_num = len(self.new_topology.tolist())
        self.degree = calculate_degree(topology)
        self.pooling_seq = pool_seq(self.degree)
        self.merge_pairs = self._get_merge_pairs()
        self.merge_nums = [len(each) for each in self.merge_pairs]

    def _get_merge_pairs(self):
      merge_pair_list = []
      for seq in self.pooling_seq:
          if len(seq) == 1:
              merge_pair_list.append([seq[0]])
          else:
              for i in range(0, len(seq), 2):
                  if i + 1 < len(seq):
                      merge_pair_list.append([seq[i], seq[i+1]])
                  else:
                      merge_pair_list.append([seq[i]])
      return merge_pair_list


    def forward(self, x):
      results = [x[:, :, 0:1, :]]
      for merge_pair in self.merge_pairs:
          merged = torch.stack([x[:, :, idx: idx+1, :] for idx in merge_pair], dim=2)
          avg = torch.mean(merged, dim=2)
          results.append(avg)
      result = torch.cat(results, dim=2)
      if result.shape[2] != self.new_joint_num:
          raise ValueError('Joint number does not match after pooling')
      return result
    

class SkeletonUnPool(nn.Module):
    def __init__(self, un_pool_expand_nums: list):
        super().__init__()
        self.un_pool_expand_nums = un_pool_expand_nums

    def forward(self, x):
        result_list = [x[:, :, 0:1, :]]  # add root joint's feature tensor first
        for idx, expand_num in enumerate(self.un_pool_expand_nums):
            tmp_idx = idx + 1
            tmp_x = x[:, :, tmp_idx: tmp_idx + 1, :].repeat(1, 1, expand_num, 1)
            result_list.append(tmp_x)
        out = torch.cat(result_list, dim=2)
        return out


def calculate_degree(topology):
    joint_num = len(topology)
    matrix = [[0] * joint_num for _ in range(joint_num)]
    for i, j in enumerate(topology.tolist()):
        matrix[i][j] = matrix[j][i] = 1
    for i in range(joint_num):
        matrix[i][i] = 0
    degree_list = [sum(row) for row in matrix]
    return degree_list

def calculate_neighbor_matrix(topology):
    joint_num = len(topology)
    mat = [[100000] * joint_num for _ in range(joint_num)]
    for i, j in enumerate(topology.tolist()):
        mat[i][j] = mat[j][i] = 1
    for i in range(joint_num):
        mat[i][i] = 0
    # Floyd-Warshall:
    for k in range(joint_num):
        for i in range(joint_num):
            for j in range(joint_num):
                mat[i][j] = min(mat[i][j], mat[i][k] + mat[k][j])
    return mat

def find_neighbor(topology, dist, ee_id):
    distance_mat = calculate_neighbor_matrix(topology)
    neighbor_list = [[] for _ in range(len(distance_mat))]
    for i in range(len(distance_mat)):
        for j in range(len(distance_mat)):
            if distance_mat[i][j] <= dist:
                neighbor_list[i].append(j)

    root_neighbors = neighbor_list[0].copy()
    root_neighbors.extend(ee_id)
    root_neighbors = list(set(root_neighbors))
    if 0 not in root_neighbors:
        root_neighbors.append(0)
    neighbor_list[0] = root_neighbors

    return neighbor_list

def pool_seq(degree):
    num_joint = len(degree)
    seq_list = [[]]
    for joint_idx in range(1, num_joint):
        if degree[joint_idx] == 2:
            seq_list[-1].append(joint_idx)
        else:
            seq_list[-1].append(joint_idx)
            seq_list.append([])
            continue
    seq_list = [each for each in seq_list if len(each) != 0]
    return seq_list

def build_bone_topology(topology):
    edges = [(topology[i], i) for i in range(1, len(topology))]
    return edges