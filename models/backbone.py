# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
""" 

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding

# from ECANet.models.eca_resnet import eca_resnet50 # imported below to avoid circular import


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """
    # make training faster and more stable by normalizing activation vectors
    # frozen: stats and affine params are fixed 

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        # register buffer: open RAM for parameters not optimized during training.
        # ie. they are consistent values (ie. frozen norm)
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        
        """
        print("x.size() =", x.size())
        print("self.weight.size() =", self.weight.size())
        print("self.bias.size() =", self.bias.size())
        print("self.running_var.size() =", self.running_var.size())
        print("self.running_mean.size() =", self.running_mean.size())
        
        for x.size() = [2, C, x, y]:
        weight, bias, running_var, running_mean all have size torch.Size([C]) (ie. a C-dimensional array)
        After .reshape(1, -1, 1, 1) these become torch.Size([1, C, 1, 1]) (to agree with the batch size)

        # if we are using just one instance of it
        x.size() = torch.Size([2, 64, 320, 384])
        self.weight.size() = torch.Size([64])
        self.bias.size() = torch.Size([64])
        self.running_var.size() = torch.Size([64])
        self.running_mean.size() = torch.Size([64])
        x.size() = torch.Size([2, 64, 288, 426])
        self.weight.size() = torch.Size([64])
        self.bias.size() = torch.Size([64])
        self.running_var.size() = torch.Size([64])
        self.running_mean.size() = torch.Size([64])
        x.size() = torch.Size([2, 64, 336, 466])
        self.weight.size() = torch.Size([64])
        self.bias.size() = torch.Size([64])
        self.running_var.size() = torch.Size([64])
        self.running_mean.size() = torch.Size([64])
        """
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias # returns the layer's output by applying linear transform of the scale and bias


class BackboneBase(nn.Module):
    """ Base class for backbone (ie. a general backbone)
    """

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        
        # C = 2048
        # H,W = H0/32, W0/32        
        
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
            
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        """
        From tensor_list, which is an instance of NestedTensor:

        tensor_list.tensors.size() = torch.Size([2, 3, x, y]) (example: x,y=768,650)
        Basically, this is a size-2 batch of color images that have both been padded to the same size [3, x, y]

        tensor_list.mask.size() = torch.Size([2,x,y])
        Here are the masks which go with the associated images/tensors.
        """
        xs = self.body(tensor_list.tensors) # pass images through backbone to get feature maps # forward step 4: from eca_resnet50    
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            """ For each batch:
            xs.size() = [2,3,a,b] (see above: this is before passing into backbone)
            After passing through backbone, xs is collections.OrderedDict

            name seems to enumerate the blocks in the backbone

            it seems as though a = 8x and b ~= 8y:
            name = 0, x.size() = [2, 512, x, y], mask.size() = [2, x, y]
            name = 1, x.size() = [2, 1024, x/2, y/2], mask.size() = [2, x/2, y/2]
            name = 2, x.size() = [2, 2048, x/4, y/4], mask.size() = [2, x/4, y/4]

            For all of the above (name = 0,1,2):
            m.size() = [2, a, b] (makes sense, as it is stores 2d masks for the images, each the same size of the padded images)
            """
            m = tensor_list.mask
            assert m is not None
            """
            If m.size()=[2,a,b], m[None].size()=[1,2,a,b]. We convert boolean matrix m[None] to a float for interpolation purposes, downsample to [x, y] (or [x/2, y/2], or [x/4, y/4]) and convert back to a boolean mask. The [0] means to get rid of the extra unneeded dimension (go from [1,2,a,b] to [2,a,b])
            """
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0] # ensure the mask matches the size of the feature maps
            """ 
            Dictionary with keys of name (0, 1, 2) and the NestedTensor (x and mask) associated at that block
            """
            out[name] = NestedTensor(x, mask) # the feature map and the mask
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 eca: bool):
        norm_layer = FrozenBatchNorm2d
        
        if eca:
            from ECANet.models.eca_resnet import eca_resnet50
            backbone = eca_resnet50(num_classes=91) # use the ECA model
            # modifed to use FrozenBatchNorm2d
            # perhaps we don't want pretrained
            # dilation is not used by eca_resnet50 but I added default [False, False, True]
        else: 
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=norm_layer) # get the resnet50 backbone
            # print("type(backbone) =", type(backbone)) # torch.models.resnet.ResNet
        
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers) # BackboneBase
        # for dilation, we reduce the stride length of the last layer by factor 2
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    """ This sequentially joins the backbone with the positional embedding (which is expressed as a module)
    """
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding) # we initialize as nn.Sequential with backbone first and position_embedding second
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):

        out: List[NestedTensor] = [] # a list of nested tensors.

        """
        Consult the comments in forward() of BackboneBase for the inputs and outputs to the backbone here.
        """
        xs = self[0](tensor_list) # send the input (images?) through the backbone.

        # print("=== Output from Backbone ===")
        for name, x in sorted(xs.items()): # items() means that the dict structure of name, x becomes list of tuples
            """
            print("name =", name) # ranges from 0 to 2 with each "output from backbone"; as the batch passes through the backbone/encoder
            print("x.tensors.size() =", x.tensors.size()) # [2, 2^{9 + name}, x/2^{name}, y/2^{name}]
            print("x.mask.size() =", x.mask.size()) # [2, x/2^{name}, y/2^{name}]
            """
            out.append(x) # output of backbone

        # position encoding for the particular image (for what came through the backbone)
        pos = []
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype)) # the associated position

        return out, pos # the backbone features and the encoding


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, args.eca)
    model = Joiner(backbone, position_embedding) # place the positional embedding after the backbone
    return model
