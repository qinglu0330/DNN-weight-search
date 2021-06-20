import importlib
import pkgutil

from .module import Conv2d, Linear
from . import network
from .network.registry import MODELS


# load all the modules residing under this package directory
for finder, name, ispkg in pkgutil.iter_modules(
        network.__path__, network.__name__ + '.'):
    importlib.import_module(name)


def Network(architecture: str, num_classes: int = 1000, FLAGS=None):
    """user interface for building a registered neural network

    Parameters
    __________
    architecture: str
        Name of the neural network
    num_classes: int, optional
        Number of objet categories the network is required to
        differienate the images into
    quant_vals: list/tuple, optional
        Actual values to quantize the networks into
    n_bits: int, optional
        number of bits when quantizting for integer operation

    Returns
    _______
    nn.Module
        an instance of nn.Module subclasses
    """
    if architecture not in MODELS:
        raise TypeError(
              "architecture {} not supported.".format(architecture))
    network = MODELS[architecture]
    model = network(num_classes=num_classes, FLAGS=FLAGS)
    # if FLAGS.quant_mode != "by bit" or FLAGS.active_bit == 0:
    #     for m in model.modules():
    #         _initialize_alpha(m)
    return model


__all__ = [*MODELS.keys(), Conv2d, Linear, Network]
