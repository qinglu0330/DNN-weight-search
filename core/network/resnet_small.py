# import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from ..module import Conv2d, Linear
from .registry import register


norm_layer = nn.BatchNorm2d
nonlinear = nn.ReLU


def _weights_init(m):
    # classname = m.__class__.__name__
    if isinstance(m, (Conv2d, Linear)) and m.quantized is False:
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A',
                 quantized=False, FLAGS=None):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1,
            quantized=quantized, FLAGS=FLAGS)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1,
            quantized=quantized, FLAGS=FLAGS)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes//4, planes//4), "constant", 0)
                        )
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     Conv2d(
                         in_planes, self.expansion * planes,
                         kernel_size=1, stride=stride, bias=False,
                         quantized=quantized, FLAGS=FLAGS),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


@register
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, FLAGS=None):
        super(ResNet, self).__init__()
        self.in_planes = 16
        quantized = FLAGS.quantize_network

        self.conv1 = Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1,
            quantized=(quantized and FLAGS.quantize_first_layer),
            FLAGS=FLAGS)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(
            block, 16, num_blocks[0], stride=1,
            quantized=quantized,
            FLAGS=FLAGS)
        self.layer2 = self._make_layer(
            block, 32, num_blocks[1], stride=2,
            quantized=quantized,
            FLAGS=FLAGS)
        self.layer3 = self._make_layer(
            block, 64, num_blocks[2], stride=2,
            quantized=quantized,
            FLAGS=FLAGS)
        self.linear = Linear(
            64, num_classes,
            quantized=(quantized and FLAGS.quantize_first_layer),
            FLAGS=FLAGS)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride,
                    quantized=False, FLAGS=None):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes, planes, stride,
                    quantized=quantized, FLAGS=FLAGS)
                    )
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


@register
def resnet20(num_classes=1000, FLAGS=None):
    return ResNet(
        BasicBlock, [3, 3, 3], num_classes=num_classes, FLAGS=FLAGS)
