import math

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
    )


class SELayer(nn.Module):
    def __init__(self, planes) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1 = conv1x1(planes, planes // 16)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(planes // 16, planes)

    def forward(self, x):
        w = self.gap(x)
        w = self.conv1(w)
        w = self.relu(w)
        w = self.conv2(w).sigmoid()
        out = x * w
        return out


class BasicBlock(nn.Module):
    """ResNet basic block."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, se=False) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = se

        if self.se:
            self.selayer = SELayer(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.se:
            out = self.selayer(out)

        out = out + residual
        out = self.relu(out)

        return out


class SEResNet4(nn.Module):
    """SE-ResNet with 4 layers."""

    def __init__(self, block, layers, inplanes=None, se=False) -> None:
        """
        layers : array of total blocks per-layer. e.g = [2 2 2 2] means that there are 4 layers, each consists of 2 blocks
        """
        super().__init__()

        if inplanes is None:  # can be referenced from the output of 3D-CNN
            self.inplanes = 64
        else:
            self.inplanes = inplanes

        self.se = se

        # create layers
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(
            inplanes * 8
        )  # inferred from inplanes of the last layer, using 1D-BN

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = (
                    m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                )  # total number of weight of each module
                m.weight.data.normal_(0, math.sqrt(2.0 / n))  # Kaiming initialization
            elif isinstance(m, nn.BatchNorm2d) or isinstance(
                m, nn.BatchNorm1d
            ):  # ```or``` support lazy evaluation
                m.weight.data.fill_(1)  # set all weight 1
                m.bias.data.zero_()  # set bias to zero

    def _make_layer(self, block, planes, blocks, stride=1):
        """Create ResNet Layer."""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)  # reshape to match BN-1D input
        out = self.bn(out)

        return out


class VideoSEResNet4(nn.Module):
    """3D-SE-ResNet with 4 layers."""

    def __init__(self, in3d=1, out3d=64, se=False) -> None:
        """
        in3d : #channel of input video
        out3d : #channel of 3d-conv output
        """
        super().__init__()

        self.out3d = out3d

        # 3D-conv frontend
        self.frontend3d = nn.Sequential(
            nn.Conv3d(
                in3d, out3d, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False
            ),
            nn.BatchNorm3d(out3d),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        # resnet
        self.resnet = SEResNet4(BasicBlock, [2, 2, 2, 2], inplanes=out3d, se=se)
        self.dropout = nn.Dropout(p=0.5)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = (
                    m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                )  # total number of weights in the module
                m.weight.data.normal_(0, math.sqrt(2.0 / n))  # Kaiming initialization
                if m.bias is not None:
                    m.bias.data.zero_()  # zero bias

            elif isinstance(m, nn.Conv2d):
                n = (
                    m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                )  # total number of weights in the module
                m.weight.data.normal_(0, math.sqrt(2.0 / n))  # Kaiming initialization
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels  # total number of weights in the module
                m.weight.data.normal_(0, math.sqrt(2.0 / n))  # Kaiming initialization
                if m.bias is not None:
                    m.bias.data.zero_()

            elif (
                isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.BatchNorm1d)
                or isinstance(m, nn.BatchNorm3d)
            ):
                m.weight.data.fill_(1)  # set all weights to 1
                m.bias.data.zero_()  # set bias to zero

    def forward(self, x):
        b, _ = x.size()[:2]  # batch size and time length

        x = x.transpose(1, 2)  # flip channel and time dimension
        x = self.frontend3d(x)
        x = x.transpose(1, 2)  # flip dimension again
        x = x.contiguous()
        x = x.view(-1, self.out3d, x.size(3), x.size(4))
        x = self.resnet(x)

        x = self.dropout(x)
        feat = x.view(b, -1, self.out3d * 8)  # last dimension must match LSTM input

        return feat
