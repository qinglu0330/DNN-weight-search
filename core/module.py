import torch
import torch.nn as nn
import torch.nn.functional as F

from .categorical import categorical_sample
from .utils import init_alpha, reset_parameters


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, quantized=False,
                 FLAGS=None):
        super().__init__()
        assert FLAGS is not None

        self.FLAGS = FLAGS
        self.quantized = quantized
        self.kh, self.kw = self.kernel_size = _pair(kernel_size)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        assert in_channels % groups == 0
        if bias is True:
            self.bias = nn.parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        _hook_parameters(self)

    def get_weight(self):
        return _get_weight(self)

    def forward(self, x):
        weight = self.get_weight()
        x = F.conv2d(
            x, weight, bias=self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation,
            groups=self.groups)
        return x

    def __str__(self):
        return "Conv2d({}, {}, kernel_size={}, stride={}, quantized={}), ".\
            format(self.in_channels, self.out_channels, self.kernel_size,
                   self.stride, self.quantized)

    def __repr__(self):
        return self.__str__()


class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, quantized=False,
                 FLAGS=None):
        super().__init__()
        assert FLAGS is not None

        self.FLAGS = FLAGS
        self.quantized = quantized
        self.in_features, self.out_features = in_features, out_features
        if bias is True:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        _hook_parameters(self)

    def get_weight(self):
        return _get_weight(self)

    def forward(self, x):
        weight = self.get_weight()
        x = F.linear(x, weight, self.bias)
        return x

    def __str__(self):
        return "Linear(in_features={}, out_features={}, quantized={})".\
            format(self.in_features, self.out_features, self.quantized)

    def __repr__(self):
        return self.__str__()


def _pair(x):
    if isinstance(x, int):
        return (x, x)
    else:
        return x


def _hook_parameters(model):
    if isinstance(model, Conv2d):
        weight_size = (model.out_channels, model.in_channels//model.groups,
                       model.kh, model.kw)
    elif isinstance(model, Linear):
        weight_size = (model.out_features, model.in_features)
    else:
        raise f"model type '{model.__class__}' not supported."

    FLAGS = model.FLAGS
    if model.quantized is False:
        model.weight = nn.Parameter(torch.randn(weight_size))
        reset_parameters(model.weight, model.bias)
        return

    if FLAGS.quant_mode == "by value":
        model.register_buffer("qvals", torch.tensor(
            FLAGS.quant_vals, requires_grad=False).float())
        alpha_size = weight_size + (len(FLAGS.quant_vals), )
        model.alpha = nn.Parameter(init_alpha(alpha_size))
        return

    if FLAGS.quant_mode != "by bit":
        raise f"quantization scheme '{FLAGS.quant_mode}' is not supported"

    model.register_buffer(
        "sign_vals", torch.tensor([-1, 1], requires_grad=False))
    model.register_buffer(
        "mag_vals", torch.tensor([0, 1], requires_grad=False))
    alpha_size = (FLAGS.num_bits, ) + weight_size + (2, )
    model.alpha = nn.Parameter(init_alpha(alpha_size))
    # setattr(model, "Bit0", nn.Parameter(torch.ones(weight_size)))
    # for i in range(1, FLAGS.num_bits):
    #     setattr(model, f"Bit{i}",
    #             nn.Parameter(torch.ones(weight_size)))

    if FLAGS.quant_format == "int":
        model.register_buffer(
            "bit_weight", 2 ** -torch.arange(
                FLAGS.num_bits - 1,
                requires_grad=False).float() / 2)

    # po2 to be added
    # mask to be added
    return


def _get_weight(model):
    FLAGS = model.FLAGS
    hard = FLAGS.hard or (model.training is False)
    temp = FLAGS.temp
    random = FLAGS.random

    if model.quantized is False:
        return model.weight

    if FLAGS.quant_mode == "by value":
        weight = categorical_sample(
            model.alpha, model.qvals, hard=hard, temp=temp,
            random=random)
        return weight

    if FLAGS.quant_format == "int":
        bit_vals = []
        for i in range(FLAGS.num_bits):
            bit_val = categorical_sample(
                model.alpha[i],
                (model.sign_vals if i == 0 else model.mag_vals),
                hard=True, temp=temp, random=random)
            if i != FLAGS.active_bit:
                bit_val = bit_val.detach()
            bit_vals.append(bit_val)
        sign_bit = bit_vals[0]
        if FLAGS.num_bits > 1:
            mag_bits = torch.stack(bit_vals[1:], dim=-1)
        else:
            mag_bits = 0
        mag_vals = (mag_bits * model.bit_weight).sum(-1, keepdim=False)
        weight = sign_bit * (mag_vals + 2 ** -(FLAGS.num_bits - 1))
        return weight
