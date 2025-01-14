import torch.nn as nn
import math
# import torch.utils.model_zoo as model_zoo
from .eca_module import eca_layer

import sys
sys.path.insert(1, "..")

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """Not used for ResNet-50"""
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, dilation=dilation)


class ECABasicBlock(nn.Module):
    """
    Not used for ResNet-50.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        from UADD.models.backbone import FrozenBatchNorm2d
        
        super(ECABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = FrozenBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = FrozenBatchNorm2d(planes)
        self.eca = eca_layer(planes, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ECABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, k_size=3):
        """Class for ECABottleneck network (the block within each conv layer)

        Args:
            inplanes (int): the number of input channels of the input feature maps (tensor)
            planes (int): the number of filters / output channels
            stride (int, optional): conv stride. Defaults to 1.
            downsample (Any, optional): used like a boolean to select downsampling. Defaults to None.
            k_size (int, optional): kernel size. Defaults to 3.
        """
        from UADD.models.backbone import FrozenBatchNorm2d
        
        super(ECABottleneck, self).__init__()
        norm_layer = FrozenBatchNorm2d
        #width = int(planes * (base_width / 64.0)) * groups
        width = planes
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.eca = eca_layer(planes * self.expansion, k_size)
        self.downsample = downsample
        self.stride = stride
        
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = FrozenBatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        # self.bn2 = FrozenBatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.bn3 = FrozenBatchNorm2d(planes * 4)
        # self.relu = nn.ReLU(inplace=True)
        # self.eca = eca_layer(planes * 4, k_size)
        # self.downsample = downsample
        # self.stride = stride
        
        self.W_ECA = None
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        """ 
        print(out.size())
        This corresponds to x in eca_layer (no size modifications are made to x in eca_layer)
        For each batch (batch_size=2), a tensor of size:
            3x [2, 256, x, y]
            4x [2, 512, x/2, y/2]
            6x [2, 1024, x/4, y/4]
            3x [2, 2048, x/8, y/8]

        ie. [N, C, H, W]
        """ 

        out = self.eca(out) # forward step 2: from eca_module
        
        #if W_ECA_in is not None:
        #    W_ECA = W_ECA = W_ECA_in
        
        #W_ECA = self.sigmoid(W_ECA)
        #self.W_ECA = W_ECA
        #print(W_ECA)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    # SHOULD WE USE 91 CLASSES (COCO)?
    def __init__(self, block, layers, num_classes=1000, k_size=[3, 3, 3, 3], replace_stride_with_dilation=[False, False, False]):
        """Class for the ResNet architecture

        Args:
            block (ECABottleneck): the efficient channel attention bottleneck
            layers (int 4-list): # of times to repeat conv 2,3,4,5
            num_classes (int, optional): number of classes. Defaults to 1000.
            k_size (list, optional): _description_. Defaults to [3, 3, 3, 3].
        """
        from UADD.models.backbone import FrozenBatchNorm2d
        
        self.groups = 1
        self.base_width = width_per_group = 64
        self.inplanes = 64
        self.dilation = 1
        
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = FrozenBatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block=block, planes=64, blocks=layers[0], k_size=int(k_size[0]))
        self.layer2 = self._make_layer(block=block, planes=128, blocks=layers[1], k_size=int(k_size[1]), stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block=block, planes=256, blocks=layers[2], k_size=int(k_size[2]), stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block=block, planes=512, blocks=layers[3], k_size=int(k_size[3]), stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #nn.AvgPool2d(7, stride=1) # doesn't seem to affect it 
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # set the initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, FrozenBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        # print("RESNET MODULES")
        # for m in self.modules():
        #   if isinstance(m, ECABottleneck):
        #     print(m)

    def _make_layer(self, block, planes, blocks, k_size, stride=1, dilate=False):
        """Make one of the conv layers in resnet (a "layer" is the chunk that the residual connection skips)

        Args:
            block (ECABottleneck): the block of conv layers (that the residual jumps over)
            planes (int): the number of filters / output channels
            blocks (int): how many times to repeat the block for this particular conv layer (conv2, conv3...)
            k_size (int): kernel size.
            stride (int, optional): conv stride. Defaults to 1.

        Returns:
            _type_: _description_
        """
        from UADD.models.backbone import FrozenBatchNorm2d
        
        downsample = None
        
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # nn.Conv2d(self.inplanes, planes * block.expansion,
                #           kernel_size=1, stride=stride, bias=False),
                conv1x1(self.inplanes, planes * block.expansion, stride),
                FrozenBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, k_size, dilation=previous_dilation))
        self.inplanes = planes * block.expansion
        
        # within a particular layer, the residual is repeated
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, k_size=k_size, dilation=self.dilation))

        from UADD.util.sequential import Sequential
        #return nn.Sequential(*layers)
        return Sequential(*layers)

    def forward(self, x):
        """Send the feature map through the ResNet.

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        x = self.conv1(x) # conv 1
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # each one of these is from forward step 3: Bottleneck
        # x, W_ECA = self.layer1(x, W_ECA)
        # x, W_ECA = self.layer2(x, W_ECA)
        # x, W_ECA = self.layer3(x, W_ECA)
        # x, W_ECA = self.layer4(x, W_ECA)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x) # average pooling
        
        x = x.view(x.size(0), -1) # equivalent to torch.flatten(x, 1) from the torchvision/resnet repo
        x = self.fc(x) # fully connected

        # ResNet x
        print("ResNet.forward() x.size() =", x.size())

        return x


def eca_resnet18(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-18 model.

    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet(ECABasicBlock, [2, 2, 2, 2], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def eca_resnet34(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-34 model.

    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet(ECABasicBlock, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def eca_resnet50(k_size=[3,3,3,3], num_classes=1000, pretrained=False):
    """Constructs a ResNet-50 model.

    Args:
        k_size: Adaptive selection of kernel size (kernel size formula says 3,5,5,5 for adaptive; 5,5,5,5 for simplicity)
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print("Constructing eca_resnet50......")
    model = ResNet(ECABottleneck, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size) # 3,4,6,3 specifies how many times to repeat conv2,3,4,5 to get 50-layer ResNet
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    return model


def eca_resnet101(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-101 model.

    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(ECABottleneck, [3, 4, 23, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def eca_resnet152(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-152 model.

    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(ECABottleneck, [3, 8, 36, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model
