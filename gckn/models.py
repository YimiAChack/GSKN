# -*- coding: utf-8 -*-
import torch
from torch import nn
# from .layers import Linear
from .layers import PathLayer, NodePooling
import numpy as np


class PathSequential(nn.Module):
    def __init__(self, input_size, hidden_sizes, path_sizes, anonymous_walk_length,
                 kernel_funcs=None, kernel_args_list=None,
                 pooling='mean', #global_pooling='sum',
                 aggregation=False, **kwargs):
        super(PathSequential, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes # 32
        
        self.path_sizes = path_sizes # 3
        self.n_layers = len(hidden_sizes)
        self.aggregation = aggregation
        self.features_anonymous_dim = anonymous_walk_length 
        self.anonymous_walk_length = anonymous_walk_length

        layers = []
        output_size = hidden_sizes[-1]
        for i in range(self.n_layers): 
            if kernel_funcs is None:
                kernel_func = "exp" 
            else:
                kernel_func = kernel_funcs[i]
            if kernel_args_list is None:
                kernel_args = 0.5  
            else:
                kernel_args = kernel_args_list[i]

            layer = PathLayer(input_size, hidden_sizes[i], path_sizes[i], self.features_anonymous_dim, anonymous_walk_length,
                              kernel_func, kernel_args, pooling, aggregation, **kwargs)
            layers.append(layer)
            input_size = hidden_sizes[i]
            if aggregation:
                output_size *= path_sizes[i]
        self.output_size = output_size

        self.layers = nn.ModuleList(layers)

    def __getitem__(self, idx):
        return self.layers[idx]

    def __len__(self):
        return self.n_layers

    def __iter__(self):
        return iter(self.layers._modules.values())

    def forward(self, features, paths_indices, anonymous_walks, other_info):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        output = features.to(device)
        paths_indices = [path_index.to(device) for path_index in paths_indices]
        for layer in self.layers:
            output = layer(output, paths_indices, anonymous_walks, other_info)
        return output

    def normalize_(self):
        for module in self.layers:
            module.normalize_()

    def unsup_train(self, data_loader, n_sampling_paths=100000, init=None, use_cuda=False):
        self.train(False)
        for i, layer in enumerate(self.layers):
            print("Training layer {}".format(i + 1))
            n_sampled_paths = 0
            n_sampled_anonymous_walks = 0
            n_sampling_anonymous_walks = n_sampling_paths
            try:
                n_paths_per_batch = (
                    n_sampling_paths + len(data_loader) - 1) // len(data_loader)
            except:
                n_paths_per_batch = 1000
                
            paths = torch.Tensor(
                n_sampling_paths, layer.path_size, layer.input_size) 
            anonymous_walks = torch.Tensor(n_sampling_paths, self.anonymous_walk_length, self.features_anonymous_dim) 

            if use_cuda:
                paths = paths.cuda()
                anonymous_walks = anonymous_walks.cuda()

            for data in data_loader.make_batch():
                if n_sampled_paths >= n_sampling_paths:
                    continue
                features = data['features']
                paths_indices = data['paths']
                n_paths = data['n_paths']
                n_nodes = data['n_nodes']
                anonymous_indices = data['anonymous_walks']
                

                if use_cuda:
                    features = features.cuda()
                    if isinstance(n_paths, list):
                        paths_indices = [p.cuda() for p in paths_indices]
                        n_paths = [p.cuda() for p in n_paths]
                    else:
                        paths_indices = paths_indices.cuda()
                        n_paths = n_paths.cuda()
                        anonymous_indices = anonymous_indices.cuda()
                    n_nodes = n_nodes.cuda()
                with torch.no_grad(): 
                    paths_batch = layer.sample_paths(
                        features, paths_indices, n_paths_per_batch)
                    anonymous_batch = layer.sample_anonymous_walks(
                        anonymous_indices, n_paths_per_batch)

                    size = min(paths_batch.shape[0], n_sampling_paths - n_sampled_paths)
                    size_anonymous = min(anonymous_batch.shape[0], n_sampling_anonymous_walks - n_sampled_anonymous_walks)
                    paths[n_sampled_paths: n_sampled_paths + size] = paths_batch[ : size]
                    anonymous_walks[n_sampled_anonymous_walks: n_sampled_anonymous_walks + size_anonymous] = anonymous_batch[ : size_anonymous]
                    n_sampled_paths += size
                    n_sampled_anonymous_walks += size_anonymous
                    
            paths = paths[:n_sampled_paths] 
            anonymous_walks = anonymous_walks[:n_sampled_anonymous_walks] 
            layer.unsup_train(paths, anonymous_walks, init=init)


class GCKNetFeature(nn.Module):
    def __init__(self, input_size, hidden_size, path_sizes, anonymous_walk_length,
                 kernel_funcs=None, kernel_args_list=None,
                 pooling='mean', global_pooling='sum',
                 aggregation=False, **kwargs):
        super().__init__()
        self.path_layers = PathSequential(
            input_size, hidden_size, path_sizes, anonymous_walk_length,
            kernel_funcs, kernel_args_list,
            pooling, aggregation, **kwargs)
        self.node_pooling = NodePooling(global_pooling)
        self.output_size = self.path_layers.output_size

    def forward(self, input, paths_indices, anonymous_walks, other_info):
        output = self.path_layers(input, paths_indices, anonymous_walks, other_info)
        return self.node_pooling(output, other_info)

    def unsup_train(self, data_loader, n_sampling_paths=100000,
                    init=None, use_cuda=False):
        self.path_layers.unsup_train(data_loader, n_sampling_paths,
                                     init, use_cuda)

    def predict(self, data_loader, use_cuda=False):
        if use_cuda:
            self.cuda()
        self.eval() # torch
        output = torch.Tensor(data_loader.n, self.output_size * 2)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        batch_start = 0
        for data in data_loader.make_batch(shuffle=False):
            features = data['features'] 
            paths_indices = data['paths']
            n_paths = data['n_paths']
            # print(n_paths)
            n_nodes = data['n_nodes']
            anonymous_walks = data['anonymous_walks']

            size = len(n_nodes)
            if use_cuda:
                features = features.cuda() 
                if isinstance(n_paths, list):
                    paths_indices = [p.cuda() for p in paths_indices]
                    n_paths = [p.cuda() for p in n_paths]
                else:
                    paths_indices = paths_indices.cuda()
                    n_paths = n_paths.cuda()
                    anonymous_walks = anonymous_walks.cuda()
                n_nodes = n_nodes.cuda()
            with torch.no_grad(): 
                features = features.to(device)
                paths_indices = [path_index.to(device) for path_index in paths_indices]
                batch_out = self.forward(features, paths_indices, anonymous_walks, {'n_paths': n_paths, 'n_nodes': n_nodes})
            output[batch_start: batch_start + size] = batch_out
            batch_start += size
        return output, data_loader.labels
