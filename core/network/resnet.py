# import torch
import torch
import torch.nn as nn
from ..module import Conv2d, Linear
from .registry import register


norm_layer = nn.BatchNorm2d
nonlinear = nn.ReLU


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1,
            quantized=False, FLAGS=None):
    """3x3 convolution with padding"""
    return Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
        groups=groups, bias=False, dilation=dilation,
        quantized=quantized, FLAGS=FLAGS
        )


def conv1x1(in_planes, out_planes, stride=1, quantized=False, FLAGS=None):
    """1x1 convolution"""
    return Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
        quantized=quantized, FLAGS=FLAGS
        )


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, quantized=False,
                 FLAGS=None):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = conv3x3(
            inplanes, planes, stride,
            quantized=quantized, FLAGS=FLAGS)
        self.bn1 = norm_layer(planes)
        self.nonlinear = nonlinear(inplace=True)
        self.conv2 = conv3x3(
            planes, planes,
            quantized=quantized, FLAGS=FLAGS)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlinear(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.nonlinear(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, quantized=False,
                 FLAGS=None):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(
            inplanes, width,
            quantized=quantized, FLAGS=FLAGS)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(
            width, width, stride, groups, dilation,
            quantized=quantized, FLAGS=FLAGS)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(
            width, planes * self.expansion,
            quantized=quantized, FLAGS=FLAGS)
        self.bn3 = norm_layer(planes * self.expansion)
        self.nonlinear = nonlinear(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlinear(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nonlinear(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.nonlinear(out)

        return out


@register
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, groups=1,
                 width_per_group=64, FLAGS=None):
        super(ResNet, self).__init__()
        quantized = FLAGS.quantize_network

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False,
            quantized=(quantized and FLAGS.quantize_first_layer),
            FLAGS=FLAGS)
        self.bn1 = norm_layer(self.inplanes)
        self.nonlinear = nonlinear(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block, 64, layers[0],
            quantized=quantized, FLAGS=FLAGS)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2,
            quantized=quantized, FLAGS=FLAGS)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2,
            quantized=quantized, FLAGS=FLAGS)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2,
            quantized=quantized, FLAGS=FLAGS)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(
            512 * block.expansion, num_classes,
            quantized=(quantized and FLAGS.quantize_last_layer),
            FLAGS=FLAGS)

        for m in self.modules():
            if isinstance(m, Conv2d) and m.quantized is False:
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
                    quantized=False, FLAGS=None):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(
                    self.inplanes, planes * block.expansion, stride,
                    quantized=quantized, FLAGS=FLAGS),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation,
                  quantized=quantized, FLAGS=FLAGS)
                  )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups,
                      base_width=self.base_width, dilation=self.dilation,
                      quantized=quantized, FLAGS=FLAGS)
                      )
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.nonlinear(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


@register
def resnet18(num_classes=1000, FLAGS=None):
    return ResNet(
        BasicBlock, [2, 2, 2, 2], num_classes=num_classes, FLAGS=FLAGS)


@register
def resnet34(num_classes=1000, FLAGS=None):
    return ResNet(
        BasicBlock, [3, 4, 6, 3], num_classes=num_classes, FLAGS=FLAGS)


@register
def resnet50(num_classes=1000, FLAGS=None):
    return ResNet(
        Bottleneck, [3, 4, 6, 3], num_classes=num_classes, FLAGS=FLAGS)
