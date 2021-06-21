import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler
import torchvision.transforms as transform


def build_data_loader(name="imagenet", path="data", train=True,
                      transform=None, distributed=False, batch_size=128,
                      num_workers=8):
    try:
        dataset = getattr(torchvision.datasets, name.upper())(
            root=path,
            transform=transform,
            train=train,
            download=True)
    except AttributeError:
        dataset = torchvision.datasets.ImageFolder(
            root=path + ("/train" if train is True else "/val"),
            transform=transform)
    if distributed is True:
        sampler = DistributedSampler(dataset, shuffle=train)
    else:
        if train is True:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, pin_memory=True)


def build_transform(transforms=[]):
    transform_list = []
    for trans in transforms:
        try:
            transform_list.append(
                transform.__dict__[trans['name']](**trans['args']))
        except KeyError:
            raise f"Operation {trans['name']} not supported!"
    return torchvision.transforms.Compose(transform_list)


class Dataset():

    def __init__(self, name='', path='', batch_size=128, num_workers=8,
                 distributed=False, transforms={}):
        self.train_loader = build_data_loader(
            name=name, path=path, train=True,
            transform=build_transform(transforms["train"]),
            distributed=distributed,
            batch_size=batch_size, num_workers=num_workers
        )
        self.val_loader = build_data_loader(
            name=name, path=path, train=False,
            transform=build_transform(transforms["val"]),
            distributed=distributed,
            batch_size=batch_size, num_workers=num_workers
        )
        self.name = name

    def __str__(self):
        return ("Dataset ({}): train set has ({}) batches and "
                "validation set has ({}) batches").format(
                    self.name, len(self.train_loader),
                    len(self.val_loader))


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


transform.Cutout = Cutout
