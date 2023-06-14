import torch
from torch import nn
from torch.nn.parameter import Parameter

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size (how many channels to use)
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # adaptive avg pool; no dimensionality reduction
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
                
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Multiply the CxHxW feature map x by the Cx1x1 channel attention (for the purpose of weighing the channels)
.
        Args:
            x (tensor): feature map

        Returns:
            tensor: feature map, with each channel scaled by that channel's attention
        """
        # feature descriptor on the global spatial information
        y = self.avg_pool(x) # GAP: global average pooling w/o dim reduction: CxHxW feature map -> 1x1xC vector of avg value per channel

        # Two different branches of ECA module
        """
        y.size() = torch.Size([2, 256, 1, 1])
        y.squeeze(-1).size() = torch.Size([2, 256, 1]) # We do the squeeze because the self avg pool made the shape [2, 256, 1, 1] and we can do away with the extra dimension for 1d conv (conv expects a 3D batched input, but this is 4d)
        y.squeeze(-1).transpose(-1, -2).size() = torch.Size([2, 1, 256]) # We do the transpose because we need to have [1, 256] rather than [256, 1] for 1d conv
        and then the transpose(-1, -2).unsqueeze(-1) brings it back to [2,256,1,1]

        # We do a 1d convolution over the vector of size [1, 256], and there are 2 of them due to batch size; without padding, it would be [2, 1, 254].
        # it is 1 input 1 output because we want to still output one vector for each single input vector, and we want the same kernel for the feature maps (assuming same channel size / stage in encoder) of all the images in the batch (and by extension, the dataset)
        """
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        """
        y.size() = 
        For each batch (batch_size=2), a tensor of size [N, C, 1, 1]:
            3x [2, 256, 1, 1]
            4x [2, 512, 1, 1]
            6x [2, 1024, 1, 1]
            3x [2, 2048, 1, 1]

        So, for each image in the batch we have 16 vectors of various lengths (1x1xC), which we will expand_as so the [1,1] becomes [x,y] to match with x for multiplication purposes.  
        """

        return x * y.expand_as(x) #, self.get_attention_weights() # ensure that the dimensions match up
        
