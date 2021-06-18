import torch
import torch.nn.functional as F


def sample_softmax(logits, hard=False, temp=1, random=0,
                   eps=1e-10, dim=-1, topk=1):
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
        # index = y_soft.max(dim, keepdim=True)[1]
        index = y_soft.topk(topk, dim=dim)[1]
        y_hard = torch.zeros_like(
            logits,
            memory_format=torch.legacy_contiguous_format
            ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def categorical_sample(logits, category_values, hard=True, temp=1,
                       random=0, topk=1):
    y = sample_softmax(logits, hard=hard, temp=temp, random=random, topk=topk)
    return (y * category_values).sum(dim=-1, keepdim=False)
