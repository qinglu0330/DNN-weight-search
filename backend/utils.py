import torch.distributed as dist


def synchronize(tensor):
    dist.barrier()
    dist.all_reduce(tensor)
    world_size = dist.get_world_size()
    return tensor.item() / world_size


def synchronize_all(*tensors):
    ret = []
    for tensor in tensors:
        ret.append(synchronize(tensor))
    return tuple(ret)
