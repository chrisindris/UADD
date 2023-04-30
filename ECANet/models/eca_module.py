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
        
        # shape of the conv1d weight
        print("CONV1D WEIGHT SHAPE")
        print(self.conv.state_dict()['weight'])
        
        self.sigmoid = nn.Sigmoid()
        
    def get_attention_weights(self):
        """getter function to show the attention weights from a particular eca_layer instance.

        Returns:
            tensor: _description_
        """
        return self.conv.state_dict()

    def forward(self, x):
        """Multiply the CxHxW feature map x by the Cx1x1 channel attention (for the purpose of weighing the channels)

        Args:
            x (tensor): feature map

        Returns:
            tensor: feature map, with each channel scaled by that channel's attention
        """
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x) # ensure that the dimensions match up
        
