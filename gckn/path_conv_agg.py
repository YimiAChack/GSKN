# -*- coding: utf-8 -*-
import torch
from gckn.gckn_fast.gckn_fast import path_conv_forward, path_conv_backward
from gckn.dynamic_pooling.pooling import dpooling_forward, dpooling_backward


MAXRAM = int(5e9)
# MAXRAM = int(100000)

def get_batch_indices(array, batch_size):
    indices = [0]
    s = 0
    for i, v in enumerate(array):
        s += v.item()
        if s > batch_size:
            indices.append(i)
            s = v.item()
    indices.append(len(array))
    return indices


from gckn.dynamic_pooling.pooling import dpooling_torch, dpooling
from gckn.gckn_fast.gckn_fast import path_conv, PathConv


def path_conv_agg(features, path_indices, kernel_size, pooling='sum', kappa=torch.exp, d_kappa=torch.exp, mask=None):
    ram_saving = MAXRAM <= (2 * path_indices.shape[0] * features.shape[-1] * features.element_size())
    if ram_saving and mask is None:
        return PathConvAggregation.apply(
            features, path_indices, kernel_size, pooling, kappa, d_kappa)
    embeded = PathConv.apply(path_indices, features)
    embeded = kappa(embeded)
    if mask is not None:
        embeded = embeded * mask.view(-1, 1)
    embeded = dpooling(embeded, kernel_size, pooling)
    return embeded
