from argparse import Namespace
from utils import read_json
import torch.distributed as dist


FLAGS = Namespace()


def setup(args):
    _copy(read_json(args.train_json))
    if args.distributed is True:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=args.local_rank)
        FLAGS.batch_per_gpu = FLAGS.batch_size // dist.get_world_size()
    else:
        FLAGS.batch_per_gpu = FLAGS.batch_size
    FLAGS.quantize_network = args.quantize_network
    FLAGS.quantize_first_layer = args.quantize_first_layer
    FLAGS.quantize_last_layer = args.quantize_last_layer
    FLAGS.hard = not args.soft
    if args.quant_json is not None:
        FLAGS.quant_mode = "by value"
        FLAGS.quant_vals = read_json(args.quant_json)
    else:
        FLAGS.quant_mode = "by bit"
        FLAGS.active_bit = args.active_bit
    return


def _copy(config_dict):
    for k, v in config_dict.items():
        setattr(FLAGS, k, v)
    return
