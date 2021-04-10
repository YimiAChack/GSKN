# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
import torch.nn.functional as F
from . import ops
from .kernels import kernels, d_kernels
from .utils import EPS, normalize_, spherical_kmeans
from .dynamic_pooling.pooling import dpooling
from .path_conv_agg import path_conv_agg

import numpy as np
from scipy import optimize
from sklearn.linear_model.base import LinearModel, LinearClassifierMixin
from sklearn.decomposition import PCA


class PathLayer(nn.Module): 
    def __init__(self, input_size, hidden_size, path_size=1, features_anonymous_dim = 5, anonymous_walk_length = 5,
                 kernel_func='exp', kernel_args=[0.5], pooling='mean',
                 aggregation=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.path_size = path_size
        
        self.features_anonymous_dim = features_anonymous_dim
        self.anonymous_walk_length = anonymous_walk_length

        self.pooling = pooling
        self.aggregation = aggregation and (path_size > 1)

        self.kernel_func = kernel_func
        if isinstance(kernel_args, (int, float)):
            kernel_args = [kernel_args]
        if kernel_func == 'exp':
            kernel_args = [1. / kernel_arg ** 2 for kernel_arg in kernel_args]
        self.kernel_args = kernel_args# [kernel_arg / path_size for kernel_arg in kernel_args]

        self.kernel_func = kernels[kernel_func]
        self.kappa = lambda x: self.kernel_func(x, *self.kernel_args)
        d_kernel_func = d_kernels[kernel_func]
        self.d_kappa = lambda x: d_kernel_func(x, *self.kernel_args)

        self._need_lintrans_computed = True
        self.weight = nn.Parameter(
            torch.Tensor(path_size, hidden_size, input_size))
        self.weight_anonym = nn.Parameter(
            torch.Tensor(anonymous_walk_length, hidden_size, self.features_anonymous_dim)) # input_size: 特征长度

        if self.aggregation:
            self.register_buffer('lintrans',
                                 torch.Tensor(path_size, hidden_size, hidden_size))
            self.register_buffer('divider',
                                 torch.arange(1., path_size + 1).view(-1, 1, 1))
        else:
            self.register_buffer('lintrans',
                                 torch.Tensor(hidden_size, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_size)
        for w in self.parameters():
            if w.dim() > 1:
                w.data.uniform_(-stdv, stdv)
        self.normalize_()

    def normalize_(self):
        normalize_(self.weight.data, dim=-1)
        normalize_(self.weight_anonym.data, dim=-1)

    def train(self, mode=True):
        super().train(mode)
        self._need_lintrans_computed = True

    def _compute_lintrans(self, weight):
        if not self._need_lintrans_computed:
            return self.lintrans
        lintrans = torch.bmm(weight, weight.permute(0, 2, 1)) 
        if self.aggregation:
            lintrans = lintrans.cumsum(dim=0) / self.divider
        else:
            lintrans = lintrans.mean(dim=0)
        lintrans = self.kappa(lintrans) 
        lintrans = ops.matrix_inverse_sqrt(lintrans) 

        if not self.training:
            self._need_lintrans_computed = False
            self.lintrans.data.copy_(lintrans.data)
        return lintrans

    
    def sample_paths(self, features, paths_indices, n_sampling_paths=1000):
        """Sample paths for a given of features and paths
        features: n_nodes x (input_path_size) x input_size
        paths_indices: n_paths x path_size
        output: n_sampling_paths x path_size x input_size
        """
        
        paths_indices = paths_indices[self.path_size - 1]

        n_all_paths = paths_indices.shape[0]
        indices = torch.randperm(n_all_paths)[ : min(n_all_paths, n_sampling_paths)] 
        paths = F.embedding(paths_indices[indices], features) 
        return paths

    def sample_anonymous_walks(self, paths, n_sampling_paths=1000):
        features = torch.from_numpy(np.eye(self.features_anonymous_dim))
        n_all_paths = paths.shape[0]
        indices = torch.randperm(n_all_paths)[ : min(n_all_paths, n_sampling_paths)]
        paths = F.embedding(paths[indices], features) 
        return paths
      

    def unsup_train(self, paths, paths_anonym, init=None):
        """Unsupervised training for path layer
        paths: n x path_size x input_size   [7442, 2, 7]
        paths_anonym: n x path_size x one_hot 
        self.weight: path_size x hidden_size x input_size
        """
        normalize_(paths, dim=-1)
        weight = spherical_kmeans(paths, self.hidden_size, init='kmeans++') # [n_clusters(hidden_size), kmer_size, n_features]

        weight = weight.permute(1, 0, 2) # [kmer_size, n_clusters, n_features] [2, 32, 7]
        self.weight.data.copy_(weight)
        
        normalize_(paths_anonym, dim=-1)
        weight_anonym = spherical_kmeans(paths_anonym, self.hidden_size, init='kmeans++') 
        weight_anonym = weight_anonym.permute(1, 0, 2) # [kmer_size, n_clusters, n_features] [2, 32, 7]
        self.weight_anonym.data.copy_(weight_anonym)


        self.normalize_()
        self._need_lintrans_computed = True

    def conv_pred(self, flag_anonymous, features, paths_indices, weight, path_size, other_info):
        norms = features.norm(dim=-1, keepdim=True) # norms: n_nodes x (input_path_size) x 1
        #output = features / norms.clamp(min=EPS)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        features = features.to(device)
        weight = weight.to(device)
        output = torch.tensordot(features, weight, dims=[[-1], [-1]]) 
        output = output / norms.clamp(min=EPS).unsqueeze(2)
        n_nodes = output.shape[0]
        if output.ndim == 4:
            output = output.permute(0, 2, 1, 3).contiguous()

        ## prepare masks
        mask = None
        if self.aggregation:
            mask = [None for _ in range(path_size)]
        if 'mask' in other_info and path_size > 1:
            mask = other_info['mask']

        output = output.view(n_nodes, path_size, -1)
        # output: n_nodes x path_size x (input_path_size x hidden_size)

        if flag_anonymous == False and self.aggregation:
            outputs = []
            for i in range(path_size):
                embeded = path_conv_agg(
                    output, paths_indices[i], other_info['n_paths'][i],
                    self.pooling, self.kappa, self.d_kappa, mask[i])
                outputs.append(embeded)
            output = torch.stack(outputs, dim=0)
            output = output.view(path_size, -1, self.hidden_size)
            # output: path_size x (n_nodes x (input_path_size)) x hidden_size
            output = norms.view(1, -1, 1) * output

            lintrans = self._compute_lintrans(weight)

            output = output.bmm(lintrans)
            output = output.permute(1, 0, 2)
            output = output.reshape(n_nodes, -1, self.hidden_size)
            output = output.contiguous()
       
        if flag_anonymous: # not aggregation
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            paths_indices = paths_indices.to(device)
            output = path_conv_agg(
                output, paths_indices, other_info['n_paths'][self.path_size - 1],
                self.pooling, self.kappa, self.d_kappa, None) 
            output = output.view(n_nodes, -1, self.hidden_size)
            output = norms.view(n_nodes, -1, 1) * output

            lintrans = self._compute_lintrans(weight)
            output = torch.tensordot(output, lintrans, dims=[[-1], [-1]])
            output = torch.squeeze(output, dim=1)
      
        return output

    def forward(self, features, paths_indices, anonymous_walks, other_info):
        """
        features: n_nodes x (input_path_size) x input_size
        paths_indices: n_paths x path_size (values < n_nodes)
        output: n_nodes x ((input_path_size) x path_size) x input_size
        """
        self.normalize_()
        output = self.conv_pred(False, features, paths_indices, self.weight, self.path_size, other_info)
        pca = PCA(n_components=self.anonymous_walk_length)
        pca.fit(anonymous_walks.reshape(features.shape[0],-1))
        features_anonymous = pca.transform(anonymous_walks.reshape(features.shape[0],-1))
        features_anonymous = torch.from_numpy(features_anonymous).float()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        features_anonymous = features_anonymous.to(device)
        output_anonymous = self.conv_pred(True, features_anonymous, anonymous_walks, self.weight_anonym, self.anonymous_walk_length, other_info)
        output = torch.cat((output, output_anonymous), 1)

        return output


class NodePooling(nn.Module):
    def __init__(self, pooling='mean'):
        super().__init__()
        self.pooling = pooling

    def forward(self, features, other_info):
        """
        features: n_nodes x (input_path_size) x input_size
        output: n_graphs x input_size
        """
        features = features.permute(0, 2, 1).contiguous()
        n_nodes = features.shape[0]
        output = dpooling(features.view(n_nodes, -1), other_info['n_nodes'], self.pooling)

        return output