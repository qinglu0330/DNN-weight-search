import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math


INIT_DIST = 'uniform'
INIT_SCALE = 1e-10
ZERO_CENTER = False
SIGN_ONLY = False
GRADIENT_TRANSFER = False
HARD = False
TEMP = 1
RANDOM = 1


def set_gradient_transfer(gradient_transfer=True):
    global GRADIENT_TRANSFER
    GRADIENT_TRANSFER = gradient_transfer
    return


def set_hard_sample(hard=True):
    global HARD
    HARD = hard
    return


def adjust_temperature(temp=1):
    global TEMP
    TEMP = temp
    return


def adjust_randomness(random=1):
    global RANDOM
    RANDOM = random
    return


def straight_through_softmax(alpha, quant_vals, hard=True, temp=1, random=1):
    y = sample_softmax(alpha, hard=hard, temp=temp, random=random)
    return (y * quant_vals).sum(dim=-1, keepdim=False)


def sample_softmax(logits, hard=False, temp=1, random=1, eps=1e-10, dim=-1):

    if not torch.jit.is_scripting():
        if type(logits) is not torch.Tensor and \
           F.has_torch_function((logits,)) is True:
            return F.handle_torch_function(
                sample_softmax, (logits,), logits, eps=eps, dim=dim)
    if eps != 1e-10:
        import warnings
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = -torch.empty_like(
        logits,
        memory_format=torch.legacy_contiguous_format).exponential_().log()
    gumbels = (logits + gumbels * random) / temp  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits,
            memory_format=torch.legacy_contiguous_format
            ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class GradientTransfer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, alpha, quant_vals):
        # y = sample_softmax(alpha)
        y = alpha
        _, max_idx = y.max(dim=-1, keepdim=False)
        weight = quant_vals[max_idx]
        ctx.save_for_backward(weight, quant_vals)
        return weight

    @staticmethod
    def backward(ctx, grad_weight):
        weight, quant_vals = ctx.saved_tensors
        n = quant_vals.size(0)
        n_dim = grad_weight.dim()
        repeat_size = (1,) * n_dim + (n,)
        grad_alpha = grad_weight.unsqueeze(-1).repeat(repeat_size)
        selected_weight = weight.unsqueeze(-1)
        collected_weights = quant_vals.view(repeat_size).expand_as(grad_alpha)
        mask = (collected_weights - selected_weight).sign()
        grad_alpha.mul_(mask)
        if SIGN_ONLY is True:
            grad_alpha.sign_()
        return grad_alpha, None


gradient_transfer_sample = GradientTransfer.apply


def categorical_sample(alpha, quant_vals, hard=False, temp=1, random=1):
    # print(GRADIENT_TRANSFER)
    if GRADIENT_TRANSFER is True:
        return gradient_transfer_sample(alpha, quant_vals)
    return straight_through_softmax(
        alpha, quant_vals, hard=hard, temp=temp, random=random)


def _pair(x):
    if isinstance(x, int):
        return (x, x)
    else:
        return x


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=False, quant_vals=None, n_bits=None, mask=False):
        super().__init__()
        self.kh, self.kw = self.kernel_size = _pair(kernel_size)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        assert in_channels % groups == 0
        # we don't quantize bias as it won't introduce multiplication
        # it should be taken care of in activation quantization process
        if bias is True:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        self.quantized = (quant_vals is not None) or (n_bits is not None)
        self.n_bits = None if quant_vals is not None else n_bits
        self.mask = mask
        if self.quantized is False:
            self.weight = nn.Parameter(
                torch.randn(
                    out_channels, in_channels//groups, self.kh, self.kw)
                )
            reset_parameters(self.weight, self.bias)
        else:
            if quant_vals is not None:
                assert isinstance(quant_vals, (list, tuple))
                self.register_buffer('quant_vals', torch.tensor(
                    quant_vals, requires_grad=False).float())
                alpha_size = (self.out_channels, self.in_channels//self.groups,
                              self.kh, self.kw, len(quant_vals))
            else:
                assert isinstance(n_bits, int) and n_bits > 0
                self.register_buffer('quant_vals', torch.tensor(
                    [0, 1], requires_grad=False).float())
                self.register_buffer('bit_vals', torch.flip(2 ** torch.arange(
                    n_bits, requires_grad=False), dims=(0,)).float())
                alpha_size = (self.out_channels, self.in_channels//self.groups,
                              self.kh, self.kw, n_bits, 2)
                if mask is True:
                    self.mask_alpha = nn.Parameter(init_alpha((n_bits, )))
                    mask_embedding = torch.triu(
                        torch.ones(n_bits, n_bits, requires_grad=False))
                    self.register_buffer("mask_embedding", mask_embedding)
                    self.register_buffer(
                        "mask_ranking",
                        torch.arange(self.n_bits, requires_grad=False) + 1
                    )
            self.alpha = nn.Parameter(init_alpha(alpha_size))

    def forward(self, x):
        weight = self.get_weight()
        x = F.conv2d(
            x, weight, bias=self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation,
            groups=self.groups)
        return x

    def quantize(self, quant_vals):
        if self.quantized is False:
            assert isinstance(quant_vals, (list, tuple))
            self.register_buffer('quant_vals', torch.tensor(
                quant_vals, requires_grad=False).float())
            alpha_size = (
                self.out_channels, self.in_channels//self.groups,
                self.kernel_size[0], self.kernel_size[1], len(quant_vals)
                )
            self.alpha = nn.Parameter(
                init_alpha(alpha_size)).to(self.weight.device)
            delattr(self, 'weight')
            self.quantized = True

    def dequantize(self):
        if self.quantized is True:
            self.weight = nn.Parameter(
                    torch.randn(
                        self.out_channels, self.in_channels//self.groups,
                        self.kernel_size[0], self.kernel_size[1])
                        ).to(self.alpha.device)
            reset_parameters(self.weight, self.bias)
            delattr(self, 'alpha')
            delattr(self, 'quant_vals')
            self.quantized = False

    def __str__(self):
        return "Conv2d({}, {}, kernel_size={}, stride={}, quant_vals={}), ".\
            format(self.in_channels, self.out_channels, self.kernel_size,
                   self.stride, (None if self.quantized is False else
                                 list(self.quant_vals.detach().numpy())))

    def __repr__(self):
        return self.__str__()

    def get_weight(self):
        if self.quantized is False:
            return self.weight
        else:
            weight = categorical_sample(
                self.alpha, self.quant_vals,
                hard=True if self.n_bits else HARD,
                temp=TEMP, random=RANDOM)
            if self.n_bits is not None:
                if self.mask is True:
                    mask_sample = sample_softmax(
                        self.mask_alpha, hard=True, random=0)
                    mask = (self.mask_embedding * mask_sample).sum(dim=-1)
                    weight = weight * mask
                weight = (weight * self.bit_vals).sum(
                    -1, keepdim=False) / 2 ** (self.n_bits - 1) - 1
            return weight

    def size(self):
        num_bias = 0 if self.bias is None else 1
        num_weights = (self.kw * self.kh * self.in_channels + num_bias) * \
            self.out_channels
        if self.quantized is False:
            return num_weights * 32
        else:
            if self.n_bits is None:
                return num_weights * math.log2(self.quant_vals.nelement())
            elif self.mask is True:
                mask_sample = sample_softmax(
                    self.mask_alpha, hard=True, random=0)
                bits = (mask_sample * self.mask_ranking).sum()
                return bits * num_weights
            else:
                return self.n_bits * num_weights

    def get_quantization(self):
        if self.quantized is False:
            return "FP32"
        elif self.n_bits is None:
            return f"value set: {self.quant_vals.tolist()}"
        elif self.mask is False:
            return f"{self.n_bits} bits"
        else:
            with torch.no_grad():
                mask_sample = sample_softmax(
                    self.mask_alpha, hard=True, random=0)
            return "{} bits".format(
                torch.nonzero(mask_sample == 1, as_tuple=True)[0].item() + 1)


class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, quant_vals=None,
                 n_bits=None, mask=False):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        if bias is True:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        self.quantized = (quant_vals is not None) or (n_bits is not None)
        self.n_bits = None if quant_vals is not None else n_bits
        self.mask = mask
        if self.quantized is False:
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            reset_parameters(self.weight, self.bias)
        else:
            if quant_vals is not None:
                assert isinstance(quant_vals, (list, tuple))
                self.register_buffer('quant_vals', torch.tensor(
                    quant_vals, requires_grad=False).float())
                alpha_size = (self.out_features, self.in_features,
                              len(quant_vals))
            else:
                assert isinstance(n_bits, int) and n_bits > 0
                self.register_buffer('quant_vals', torch.tensor(
                    [0, 1], requires_grad=False).float())
                self.register_buffer('bit_vals', torch.flip(2 ** torch.arange(
                    n_bits, requires_grad=False), dims=(0,)).float())
                alpha_size = (self.out_features, self.in_features, n_bits, 2)
                if mask is True:
                    self.mask_alpha = nn.Parameter(init_alpha((n_bits, )))
                    mask_embedding = torch.triu(
                        torch.ones(n_bits, n_bits, requires_grad=False))
                    self.register_buffer("mask_embedding", mask_embedding)
                    self.register_buffer(
                        "mask_ranking",
                        torch.arange(self.n_bits, requires_grad=False) + 1
                    )
            self.alpha = nn.Parameter(init_alpha(alpha_size))

    def forward(self, x):
        weight = self.get_weight()
        x = F.linear(x, weight, self.bias)
        return x

    def quantize(self, quant_vals):
        if self.quantized is False:
            assert isinstance(quant_vals, (list, tuple))
            self.register_buffer('quant_vals', torch.tensor(
                quant_vals, requires_grad=False).float())
            alpha_size = (self.out_features, self.in_features, len(quant_vals))
            self.alpha = nn.Parameter(init_alpha(alpha_size))
            delattr(self, 'weight')
            self.quantized = True

    def dequantize(self):
        if self.quantized is True:
            self.weight = nn.Parameter(
                    torch.randn(self.out_features, self.in_features)
                        ).to(self.alpha.device)
            reset_parameters(self.weight, self.bias)
            delattr(self, 'alpha')
            delattr(self, 'quant_vals')
            self.quantized = False

    def __str__(self):
        bias = (self.bias is not None)
        quant_vals = None if self.quantized is False else \
            list(self.quant_vals.detach().numpy())
        s = "Linear(in_features={}, out_features={}, bias={}, quant_vals={})".\
            format(self.in_features, self.out_features, bias, quant_vals)
        return s

    def __repr__(self):
        return self.__str__()

    def get_weight(self):
        if self.quantized is False:
            return self.weight
        else:
            weight = categorical_sample(
                self.alpha, self.quant_vals,
                hard=True if self.n_bits else HARD,
                temp=TEMP, random=RANDOM)
            if self.n_bits is not None:
                if self.mask is True:
                    mask_sample = sample_softmax(
                        self.mask_alpha, hard=True, random=0)
                    mask = (self.mask_embedding * mask_sample).sum(dim=-1)
                    weight = weight * mask
                weight = (weight * self.bit_vals).sum(
                    -1, keepdim=False) / 2 ** (self.n_bits - 1) - 1
            return weight

    def size(self):
        num_bias = 0 if self.bias is None else 1
        num_weights = (self.in_features + num_bias) * self.out_features
        if self.quantized is False:
            return num_weights * 32
        else:
            if self.n_bits is None:
                return num_weights * math.log2(self.quant_vals.nelement())
            elif self.mask is True:
                mask_sample = sample_softmax(
                    self.mask_alpha, hard=True, random=0)
                bits = (mask_sample * self.mask_ranking).sum()
                return bits * num_weights
            else:
                return self.n_bits * num_weights

    def get_quantization(self):
        if self.quantized is False:
            return "FP32"
        elif self.n_bits is None:
            return f"value set: {self.quant_vals.tolist()}"
        elif self.mask is False:
            return f"{self.n_bits} bits"
        else:
            with torch.no_grad():
                mask_sample = sample_softmax(
                    self.mask_alpha, hard=True, random=0)
            return "{} bits".format(
                torch.nonzero(mask_sample == 1, as_tuple=True)[0].item() + 1)


def reset_parameters(weight, bias=None):
    init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(bias, -bound, bound)


def init_alpha(size):
    # return evenly_init_alpha(size)
    alpha = torch.ones(size, requires_grad=False) / size[-1]
    # alpha.div_(alpha.sum(-1, keepdim=True))
    return torch.log(alpha) + \
        1e-5 * (torch.rand(size, requires_grad=False) - 0.5)
