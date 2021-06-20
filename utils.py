import matplotlib.pyplot as plt
import math
import json
# import logging
import os
import pandas as pd
import torch
import torch.distributed as dist
from core import Conv2d, Linear


def save_args(args, filename, verbose=False):
    if verbose is True:
        print(f"saving args to {filename}")
    with open(filename, 'w') as f:
        for k, v, in vars(args).items():
            f.write(str(k) + '\t' + str(v) + '\n')
    return


def display_args(args):
    for k, v in vars(args).items():
        print(f"{k}:\t\t\t\t{v}")
    return


def save_result(result, dir, verbose=False):
    data = pd.DataFrame()
    for name in result._fields:
        data[name] = getattr(result, name)
    result_path = os.path.join(dir, "result.csv")
    if verbose is True:
        print(f"saving result to {result_path}")
    data.to_csv(result_path, index=False)
    plot_result(data, dir, verbose)
    return


def save_quantization(model_quantization, dir, verbose=True):
    path = os.path.join(dir, "quantization.json")
    with open(path, 'w') as f:
        json.dump(model_quantization, f, indent=4)
    if verbose is True:
        print(f"saving quantization to {path}")
    return


def plot_result(result, dir, verbose=True):
    val = result.loc[
        :, result.columns.map(lambda s: "acc" in s)]
    # ax = val.plot()
    # fig = ax.get_figure()
    plt.figure()
    plt.plot(val)
    plt.legend([
        f"{name}: {best:.2%}" for name, best in zip(
            val.columns, val.max())
    ])
    fig_path = os.path.join(dir, 'validation result.png')
    plt.savefig(fig_path)
    return


def read_json(path):
    f = open(path, 'r')
    return json.load(f)


def count_quant_layers(model):
    count = 0
    for n, m in model.named_modules():
        if isinstance(m, (Conv2d, Linear)) and m.quantized is True:
            count += 1
    return count


def save_state(model_to_save, optimizer, checkpoint, dir, verbose=True):
    path = os.path.join(dir, "state.pth.tar")
    if verbose is True:
        print(f"saving states to {path}")
    torch.save(
        {"model_state": model_to_save.state_dict(),
         "optimizer_state": optimizer.state_dict(),
         "checkpoint": checkpoint},
        path)


def synchronize(tensor):
    dist.barrier()
    dist.all_reduce(tensor)
    world_size = dist.get_world_size()
    return tensor.item() / world_size


def compute_model_size(model):
    count = 0
    for m in model.modules():
        if isinstance(m, (Conv2d, Linear)):
            with torch.no_grad():
                count += m.get_size()
    return count


def quantization_profile(model):
    ret = {"model size": f"{compute_model_size(model) / 1e6} Mbits"}
    for n, m in model.named_modules():
        if isinstance(m, (Conv2d, Linear)):
            ret[n] = m.get_quantization()
    return ret


def sigmoid(x):
    x = min(max(x, -100), 100)
    return 1 / (1 + math.exp(-x))


def print_factory_ddp(local_rank=None):
    def print_func(*args, **kwargs):
        if not local_rank:
            print(*args, **kwargs)
        return
    return print_func


def save_model_state(model, path):
    state_dict = {}
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            state_dict[n] = m.state_dict()
        if isinstance(m, (Conv2d, Linear)):
            state_dict[n] = {
                "weight": m.get_weight(),
                "bias": m.bias
            }
    torch.save(state_dict, path)
    return


def load_model_state(model, path):
    state_dict = torch.load(path)
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.load_state_dict(state_dict[n])

        if isinstance(m, (Conv2d, Linear)):
            if m.bias is not None:
                m.bias.data.copy_(state_dict[n]["bias"].data)
            if hasattr(m, "weight"):
                m.weight.data.copy_(state_dict[n]["weight"].data)
            elif hasattr(m, "init_weight"):
                m.init_weight.data.copy_(state_dict[n]["weight"].data)
    return
