import math
import torch
import torch.nn.init as init


def init_alpha(size, requires_grad=True):
    alpha = torch.ones(size, requires_grad=requires_grad) / size[-1]
    noise = 1e-5 * (torch.rand(size, requires_grad=requires_grad) - 0.5)
    return torch.log(alpha) + noise


def reset_parameters(weight, bias=None):
    init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(bias, -bound, bound)
