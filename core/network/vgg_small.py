import torch
import torch.nn as nn
from ..module import Conv2d, Linear
from .registry import register


norm_layer = nn.BatchNorm2d
nonlinear = nn.ReLU


@register
class VGGSmall(nn.Module):

    def __init__(self, num_classes=10, FLAGS=None):
        super(VGGSmall, self).__init__()
        quantized = FLAGS.quantize_network
        # print(quantized)
        self.nonlinear = nonlinear(inplace=True)
        self.conv1 = Conv2d(
            3, 128, 3, padding=1,
            quantized=(quantized and FLAGS.quantize_first_layer),
            FLAGS=FLAGS)
        self.bn1 = norm_layer(128)
        self.conv2 = Conv2d(
            128, 128, 3,
            padding=1, quantized=quantized, FLAGS=FLAGS)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = norm_layer(128)
        self.conv3 = Conv2d(
            128, 256, 3,
            padding=1, quantized=quantized, FLAGS=FLAGS)
        self.bn3 = norm_layer(256)
        self.conv4 = Conv2d(
            256, 256, 3,
            padding=1, quantized=quantized, FLAGS=FLAGS)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn4 = norm_layer(256)
        self.conv5 = Conv2d(
            256, 512, 3,
            padding=1, quantized=quantized, FLAGS=FLAGS)
        self.bn5 = norm_layer(512)
        self.conv6 = Conv2d(
            512, 512, 3,
            padding=1, quantized=quantized, FLAGS=FLAGS)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn6 = norm_layer(512)
        self.fc = Linear(
            512 * 4 * 4, num_classes,
            quantized=(quantized and FLAGS.quantize_last_layer),
            FLAGS=FLAGS)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.nonlinear(x)
        # x = nn.functional.dropout(x, p=0.4)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.nonlinear(x)
        # x = nn.functional.dropout(x, p=0.4)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.nonlinear(x)
        # x = nn.functional.dropout(x, p=0.4)

        x = self.conv4(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.nonlinear(x)
        # x = nn.functional.dropout(x, p=0.4)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.nonlinear(x)
        # x = nn.functional.dropout(x, p=0.4)

        x = self.conv6(x)
        x = self.pool6(x)
        x = self.bn6(x)
        x = self.nonlinear(x)
        # x = nn.functional.dropout(x, p=0.4)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d) and m.quantized is False:
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, Linear) and m.quantized is False:
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


@register
def vgg_small(num_classes=10, FLAGS=None):
    return VGGSmall(num_classes=num_classes, FLAGS=FLAGS)
