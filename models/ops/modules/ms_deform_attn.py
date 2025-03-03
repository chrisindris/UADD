# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Here is where the deformable attention module is defined.

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension (indris: =C in paper)
        :param n_levels     number of feature levels (indris: =L in paper; how many output feature maps to use)
        :param n_heads      number of attention heads (indris: =M in paper)
        :param n_points     number of sampling points per attention head per feature level (indris: =K in paper; how many points around the reference point to add)
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads # C_v in paper
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # from the query feature (size dHW)
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2) # 2MK channels for sampling offsets per level
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points) # MK channels for attention weights per level
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model) # for aggregating the different heads' output

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        
        # .view() part: for each head, we have the cos term and sin term (2 values)
        # .repeat() part: repeat previous for each feature map level and deform attn offset point
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            
        # reset the weight matrices
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape # encoder: [B=2, S, d_model=256], which is size dBHW; decoder: [B=2, num queries = 300, d_model=256]
        N, Len_in, _ = input_flatten.shape # [B=2, S, d_model=256]
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in # input_spatial_shapes.shape = [4, 2] (the height/width shape of the 4 feature levels); # len_in is all of their pixels combnined

        # -- split the 256 dimension into heads, feature levels and attention points (incl. their height/width dimensions) --
        value = self.value_proj(input_flatten) # [2, S, 256]
        if input_padding_mask is not None:
            # input_padding_mask.shape = [2, S]
            value = value.masked_fill(input_padding_mask[..., None], float(0)) # [..., None] makes it [2, S, 1], broadcasted to each of the dimensions [2, S, 256]; zeros-out the parts of the image outside the mask
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads) # [2, S, num_heads=8, 32] (this splits the d_model dimension into the 8 heads)
        
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2) # [2, S, 8, 32] -> [2, S, num_heads=8, num feature levels=4, num of attention points = 4, 2 (height/width of each attention point)]; decoder does [2, 300, 8, 32] -> [2, 300, 8, 4, 4, 2]
        
        # here are the attention weights; we want to add the ECA attention to the levels dimension
        
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points) # [2, S, 128] -> [2, S, 8, 16] by view(); decoder: [2, 300, 128] -> [2, 300, 8, 16] by view()
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points) # [2, S, 8, 4, 4]; decoder [2, 300, 8, 4, 4]
        
        # N, Len_q, n_heads, n_levels, n_points, 2
        # reference_points.shape = [2, S, 4, 2]; decoder [2, 300, 4, 2]
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1) # [4, 2]
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :] # [2, S/300, 1, 4, 1, 2] + [2, S/300, 8, 4, 4, 2]/[1, 1, 1, 4, 1, 2] -> [2, S/300, 8, 4, 4, 2]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step) # [2, S/300, 256]
        output = self.output_proj(output) # [2, S/300, 256]
        return output
