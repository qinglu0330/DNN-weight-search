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

    assert FLAGS.num_bits > 1
    # model.register_buffer(
    #     "sign_vals", torch.tensor([-1, 1], requires_grad=False))
    alpha_size = weight_size + (2, )
    model.alpha = nn.Parameter(torch.log(torch.ones(alpha_size) / 2))

    if FLAGS.quant_format == "int":
        offset = 2 ** -(FLAGS.num_bits - 1)
        model.register_buffer(
            "init_weight",
            torch.ones(weight_size, requires_grad=False) - offset
        )
        model.register_buffer(
            "active_bit_weight",
            torch.tensor(
                [0, 2 ** (-FLAGS.active_bit)] if FLAGS.active_bit > 0
                else [-1, 1],
                requires_grad=False)
        )
    elif FLAGS.quant_format == "po2":
        model.register_buffer(
            "init_weight",
            torch.ones(weight_size, requires_grad=False)
        )
        model.register_buffer(
            "active_bit_weight",
            torch.tensor(
                [1, 2 ** -FLAGS.active_bit] if FLAGS.active_bit > 0
                else [-1, 1],
                requires_grad=False)
        )
    if FLAGS.quant_mode != "by bit" or FLAGS.active_bit == 0:
        _initialize_alpha(model)
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

    bit_val = categorical_sample(
        model.alpha,
        model.active_bit_weight,
        hard=True, temp=temp, random=random)
    if FLAGS.active_bit == 0:
        return bit_val * model.init_weight

    if FLAGS.quant_format == "int":
        return model.init_weight - bit_val * model.init_weight.sign()
    elif FLAGS.quant_format == "po2":
        return model.init_weight * bit_val


def _initialize_alpha(model):
    if hasattr(model, "alpha"):
        model.alpha.data.copy_(init_alpha(model.alpha.size()))
    return
