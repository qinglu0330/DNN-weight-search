from argparse import Namespace
from utils import read_json
import torch.distributed as dist


FLAGS = Namespace()


def setup(args):

    if args.distributed is True:
        FLAGS.batch_per_gpu = FLAGS.batch_size // dist.get_world_size()
    else:
        FLAGS.batch_per_gpu = FLAGS.batch_size
    _copy(read_json(args.train_json))
    _copy(read_json(args.quant_json))
    FLAGS.quant_first_layer = args.quant_first_layer
    FLAGS.quant_last_layer = args.quant_last_layer
    FLAGS.hard = args.hard
    return


def _copy(config_dict):
    for k, v in config_dict.items():
        setattr(FLAGS, k, v)
    return


def get_model():
    pass


def get_optimizer():
    pass


def get_lr_scheduler():
    pass
