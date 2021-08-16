import torch
from torch.optim.optimizer import Optimizer, required
from .module import Conv2d, Linear


class QuantSGD(Optimizer):
    """The official implementation of SGD optimizer modified by
        separating the real parameters and the proximal parameters
    """

    def __init__(self, model, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, update_alpha=True):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        proximal_params = set()
        for m in model.modules():
            if isinstance(m, (Conv2d, Linear)):
                if m.quantized is True:
                    for param in m.parameters():
                        proximal_params.add(param)
        real_params, proxy_params = [], []
        for p in model.parameters():
            if p not in proximal_params:
                real_params.append(p)
            else:
                proxy_params.append(p)
        self._count = (len(real_params), len(proxy_params))
        parameters = [
            {'params': real_params},
            {'params': proxy_params}
            ]
        self.update_alpha = update_alpha
        super(QuantSGD, self).__init__(parameters, defaults)

    def __setstate__(self, state):
        super(QuantSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        self._float_step()
        if self.update_alpha:
            self._quant_step()
        return loss

    def _float_step(self):
        group = self.param_groups[0]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        for p in group['params']:
            if p.grad is None:
                continue
            d_p = p.grad.data
            if weight_decay != 0:
                d_p.add_(weight_decay, p.data)
            if momentum != 0:
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = \
                        torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf

            p.data.add_(-group['lr'], d_p)

    def _quant_step(self):
        group = self.param_groups[1]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        for p in group['params']:
            if p.grad is None:
                continue
            d_p = p.grad.data
            if True:
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = \
                            torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
            p.data.add_(-group['lr'], d_p)

    # def _normalize(self):
    #     for alpha in self.param_groups[1]["params"]:
    #         min_alpha, _ = alpha.min(dim=-1, keepdim=True)
    #         max_alpha, _ = alpha.max(dim=-1, keepdim=True)
    #         print((alpha - min_alpha) / (max_alpha - min_alpha))

    def unquantized_layers(self):
        return self._count[0]

    def quantized_layers(self):
        return self._count[1]
