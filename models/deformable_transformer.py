# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

# Builds the deformable transformer.

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        """Initialization of the class.

        Args:
            d_model (int, optional): _description_. Defaults to 256.
            nhead (int, optional): _description_. Defaults to 8.
            num_encoder_layers (int, optional): _description_. Defaults to 6.
            num_decoder_layers (int, optional): _description_. Defaults to 6.
            dim_feedforward (int, optional): _description_. Defaults to 1024.
            dropout (float, optional): _description_. Defaults to 0.1.
            activation (str, optional): _description_. Defaults to "relu".
            return_intermediate_dec (bool, optional): _description_. Defaults to False.
            num_feature_levels (int, optional): _description_. Defaults to 4.
            dec_n_points (int, optional): _description_. Defaults to 4.
            enc_n_points (int, optional): _description_. Defaults to 4.
            two_stage (bool, optional): _description_. Defaults to False.
            two_stage_num_proposals (int, optional): _description_. Defaults to 300.
        """
        super().__init__()

        self.d_model = d_model # all multi-scale feature maps have d_model channels.
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model)) # level embedding + trained jointly with network

        if two_stage:
            # Stage 1: encoder generates region proposals
            # Stage 2: decoder refines these object queries
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            # If no proposals, we use reference points
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters(): # something like a uniform initialization
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules(): # reset the deformable attention weights
            if isinstance(m, MSDeformAttn):
                m._reset_parameters() # implemented in MSDeformAttn
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed) # scale-level embeddings are randomly initialized

    def get_proposal_pos_embed(self, proposals):
        """For two-stage; get the positional embedding of the stage 1 proposals

        Args:
            proposals (_type_): _description_

        Returns:
            _type_: _description_
        """
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device) # size 128 vector, 0 to 127
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4 # N=count per layer, L is layers, 4 because bbox
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t # None adds a new axis, of size 128 due to dimension
        # N, L, 4, 64, 2 # 64 are sin, 64 are cos
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes): # This is the shape of the object
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        """Get the ratio of the image size which is actually useful data (ie. ratio of size of original to size of padded)

        Args:
            mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        
        assert self.two_stage or query_embed is not None # for one stage, we have object query embeddings

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)): # iterate over the 4 feature levels
            bs, c, h, w = src.shape # batch: batchsize (num of imgs), channels=3, h and w for largest image
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2) # [B=2, C=256, x', y'] -> [B=2, f = x' * y', C=256] (flatten each channel to 1D)
            mask = mask.flatten(1) # [B, x', y'] -> [B, f]
            pos_embed = pos_embed.flatten(2).transpose(1, 2) # [B=2, C=256, x', y'] -> [B=2, f = x' * y', C=256]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1) # position and level embed added together; self.level_embed[lvl] is size [256] -> [1, 1, 256] by view -> broadcasted to [B, f, 256] (size of pos_embed and lvl_pos_embed)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
            
        src_flatten = torch.cat(src_flatten, 1) # concat src_flatten (a list of feature maps) into tensor of size [B, S = sum(x' * y') for the 4 feature maps, C]
        mask_flatten = torch.cat(mask_flatten, 1) # concat mask_flatten (a list of masks) into tensor of size [B, S]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # flatten the embeddings into tensor of size [B, S, C]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device) # convert list of dimension tuples to a tensor of size [4, 2]=[num_feature_levels, 2 because spatial shape is size 2 (ie. height & width)]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) # to index where the levels are of the feature maps; size [4]=[num_feature_levels] (a 4-vector due to the 4 feature maps used)
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1) # size [B=2, num_feature_levels=4, 2 because spatial shape is size 2 (ie. height & width)]

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten) # memory.size() = [B=2, S, C=256]

        # prepare input for decoder
        bs, _, c = memory.shape # batchsize, S, channel
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points # two-stage, so these are (4D, since 2 2D points) bounding boxes
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1) # splits query_embed: torch.Size([300, 512]) -> query_embed: torch.Size([300, 256]) and tgt: torch.Size([300, 256]); splitting into query and target
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1) # [300, 256] -> [B=2, 300, 256] (we copy the query embedding for all images in the batch)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1) # [300, 256] -> [2, 300, 256] (we copy the target embedding for all images in the batch)
            reference_points = self.reference_points(query_embed).sigmoid() # through a learned linear projection, query_embed:[B=2, 300, 256] -> reference_points:[B=2, 300, 2 since each point is 2D (height & width)] (each value squeezed to range [0, 1] via sigmoid)
            init_reference_out = reference_points # this is one-stage, so reference_points has (2D) points

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten) # target (feature vectors), reference points
        # hs.size() = torch.Size([6, 2, 300, 256]; inter_references = torch.Size([6, 2, 300, 2])

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024, # dim of hidden layer of ffn
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """For adding the level and positional embedding to the tensor

        Args:
            tensor (_type_): _description_
            pos (_type_): _description_

        Returns:
            _type_: _description_
        """
        return tensor if pos is None else tensor + pos
    

    # -- Forward: Passing the array through --

    def forward_ffn(self, src):
        """
        Apply ffn, with hidden layer of size d_ffn with dropout and layer normalization 
        
        Parameters:
            src (Tensor): The input source tensor (the batch + pos/layer embeddings).
        
        Returns:
            Tensor: The output source tensor after applying the feed-forward neural network.
        """
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src) # add and norm
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # src: the source (input) ie. the extracted features

        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask) # [2, S, 256]
        src = src + self.dropout1(src2) # all are [2, S, 256]
        src = self.norm1(src) # [2, S, 256]

        # ffn
        src = self.forward_ffn(src) # [2, S, 256]

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """The reference point is the initial guess of the box center

        Args:
            spatial_shapes (_type_): _description_
            valid_ratios (_type_): _description_
            device (_type_): _description_

        Returns:
            _type_: _description_
        """
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1) # concatenate the list into a tensor
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device) # [2, S, 4, 2]
        
        # send the src input through the layers (the N=6 encoder layer copies)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask) # all are [2, S, 256]

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # we are just making variables here, so the order doesn't matter.

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout) # non-deformable, since conv feature maps are key elements
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        """feed-forward network; connects the true target with the target predicted from the self and cross attention 

        Args:
            tgt (tensor): size [2, 300, d_model=256]

        Returns:
            tensor: of the same size
        """
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        """
        Forward pass through the model.

        Args:
            tgt (Tensor): Target bounding boxes. [2, 300, 256]
            query_pos (Tensor): Query positions for deformable attention. [2, 300, 256]
            reference_points (Tensor): Reference points for deformable attention.
            src (Tensor): Source tensors.
            src_spatial_shapes (List[Tuple[int, int]]): Spatial shapes of source tensors.
            level_start_index (List[int]): Start indices of each level in the source tensors.
            src_padding_mask (Tensor, optional): Padding mask for source tensors. Defaults to None.

        Returns:
            Tensor: Output tensor after passing through the model.
        """
        # tgt: target (the bounding boxes)
        # reference points: for deformable attention

        # self attention
        q = k = self.with_pos_embed(tgt, query_pos) # [2, 300, 256]
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1) # [2, 300, 256]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask) # [2, 300, 256]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt) # [2, 300, 256]

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers): # lid = layer index (iterates num_decoder_layers=6 times)
            if reference_points.shape[-1] == 4: # reference bounding boxes ie. 2 points
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2 # each reference point is a single (h, w) point
                """
                reference_points_input: [B=2, proposals=300, layers=4, reference_point dimensions = 2] from reference_points[:, :, None]: [2,300,1,2] and src_valid_ratios[:, None]: [2,1,4,2]
                - the indexing of None explicitly specifies an extra dimension (the 1)
                - to fit in [2, 300, 4, 2]: src_valid_ratios is copied 300 times (ie. for each proposal) and reference_points is copied 4 times (for each feature level)
                """
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask) # [2, 300, 256]

            # hack implementation for iterative bounding box refinement
            # See DefDETR A.4
            # Should the [..., :2] be put in the == 4 case?
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate: # ie. retain the output and ref points from each of the n=6 decoder layers
                intermediate.append(output) # [2, 300, 256]
                intermediate_reference_points.append(reference_points) # [2, 300, 2]

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points) # [6=num_decoder_layers, 2=batch_size, 300= num of obj. queries or obj. proposals, 256=d_model], [6=num_decoder_layers, 2=batch_size, 300= num of obj. queries or obj. proposals, 2= dimensions of each 2D reference point (h, w)]

        return output, reference_points


def _get_clones(module, N):
    """ Clones the encoder or decoder block N=6 times.
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries)


