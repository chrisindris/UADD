# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """Set the values we need for the position embedding

        Args:
            num_pos_feats (int, optional): _description_. Defaults to 64.
            temperature (int, optional): _description_. Defaults to 10000.
            normalize (bool, optional): _description_. Defaults to False.
            scale (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        """Computes the positional encoding for the input vector.

        Args:
            tensor_list (NestedTensor): _description_

        Returns:
            _type_: _description_
        """
        x = tensor_list.tensors
        mask = tensor_list.mask
        
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        """
        512, 1024, 2048, 256 for a batch?
        pos.size: reduces C by factor of 2
        
        PositionEmbeddingSine: x.size() = torch.Size([2, 512, 92, 110])
        PositionEmbeddingSine: mask.size() = torch.Size([2, 92, 110])
        PositionEmbeddingSine: pos.size() = torch.Size([2, 256, 92, 110])
        PositionEmbeddingSine: x.size() = torch.Size([2, 1024, 46, 55])
        PositionEmbeddingSine: mask.size() = torch.Size([2, 46, 55])
        PositionEmbeddingSine: pos.size() = torch.Size([2, 256, 46, 55])
        PositionEmbeddingSine: x.size() = torch.Size([2, 2048, 46, 55])
        PositionEmbeddingSine: mask.size() = torch.Size([2, 46, 55])
        PositionEmbeddingSine: pos.size() = torch.Size([2, 256, 46, 55])
        PositionEmbeddingSine: x.size() = torch.Size([2, 256, 23, 28])
        PositionEmbeddingSine: mask.size() = torch.Size([2, 23, 28])
        PositionEmbeddingSine: pos.size() = torch.Size([2, 256, 23, 28])
        """
        
        """
        print("pos.size() =", pos.size())
        
        For one batch:
        pos.size() = torch.Size([2, 256, 80, 96])
        pos.size() = torch.Size([2, 256, 40, 48])
        pos.size() = torch.Size([2, 256, 20, 24])
        pos.size() = torch.Size([2, 256, 10, 12])
        """
        
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    Not used by default.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'): # default is sine
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
