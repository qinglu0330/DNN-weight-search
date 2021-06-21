#!/usr/bin/env python
import argparse
import collections
import datetime
import os
import time

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
import warnings


from backend import Dataset, train_epoch, evaluate
from core import Network
from common import setup, FLAGS
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    # Fundamental arguments: the dataset and network architecture
    parser.add_argument('-a', '--arch', type=str, default='vgg_small')
    parser.add_argument('--load-model-state', type=str, default=None)
    parser.add_argument('-d', '--dataset', type=str, default="cifar10")
    parser.add_argument('-p', '--path', type=str, default='data/')
    parser.add_argument('--transform-json', type=str, default=None)
    parser.add_argument('--num-classes', type=int, default=1000)

    # Quantization
    parser.add_argument('-q', '--quantize-network', action="store_true")
    parser.add_argument('--quant-json', type=str, default=None)
    parser.add_argument('--quantize-first-layer', action='store_true')
    parser.add_argument('--quantize-last-layer', action='store_true')
    parser.add_argument('--soft', action='store_true', help="sample stratergy")
    parser.add_argument('--active-bit', type=int, default=0)

    # training configuration
    parser.add_argument('--train-json', type=str, default=None)

    # Arguments for training logistics
    parser.add_argument('--sync-bn', action='store_true')
    parser.add_argument('--local_rank', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0, metavar='S')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('-o', '--work-dir', type=str, default="results/")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help="turn on the verbose mode")
    args = parser.parse_args()
    return args


args = parse_args()
warnings.filterwarnings("ignore")
torch.manual_seed(args.seed)
cudnn.benchmark = True
print = utils.print_factory_ddp(args.local_rank)


def main():
    args.distributed = True if args.local_rank is not None else False
    args.verbose = args.verbose and (not args.local_rank)
    setup(args)
    print('*' * 40 + "Setup Details" + '*' * 40)
    for k, v in vars(FLAGS).items():
        print(f"{k}:\t\t\t\t{v}")
    time.sleep(5)

    print('*' * 40 + "start" + '*' * 40)
    print("creating dataset ===>")
    args.transform_json = args.transform_json or \
        os.path.join(f"json/transform/{args.dataset.lower()}.json")
    dataset = Dataset(
        args.dataset, args.path,
        batch_size=FLAGS.batch_per_gpu,
        transforms=utils.read_json(args.transform_json),
        distributed=args.distributed)

    print("creating model ===>")
    device = torch.device(
        f"cuda:{args.local_rank if args.distributed is True else args.gpu}")
    model = Network(
        args.arch,
        num_classes=args.num_classes, FLAGS=FLAGS
        ).to(device)
    if args.load_model_state:
        # model.load_state_dict(torch.load(args.load_model_state))
        utils.load_model_state(model, args.load_model_state)
        print("Loading successful, evaluating the model")
        loss, acc1, acc5, _ = evaluate(
            model, dataset.val_loader,
            verbose=args.verbose,
            device=device, distributed=args.distributed)
        print(f"Eval result => Averaged loss: {loss:.5f}, acc@1: {acc1:.2%}, "
              f"acc@5: {acc5:.2%} ")

    num_quant_layers = utils.count_quant_layers(model)
    print(f"{args.arch}: {num_quant_layers} layers are quantized")
    if args.distributed is True:
        if FLAGS.sync_bn is True:
            print("synchronizing BN")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model, device_ids=[device], output_device=device
            )
    else:
        print("Mode: single-node single-gpu")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=FLAGS.init_lr,
        momentum=FLAGS.momentum,
        weight_decay=FLAGS.weight_decay)

    if FLAGS.optimizer == "SGD":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=FLAGS.milestones, gamma=FLAGS.gamma
        )
    elif FLAGS.optimizer == "Cos":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, FLAGS.epochs
        )
    else:
        raise f"optimizer type {FLAGS.optimizer} not known"

    result = fitting(
        model, dataset, optimizer, lr_scheduler,
        epochs=FLAGS.epochs, device=device, verbose=args.verbose,
        distributed=args.distributed)

    if not args.local_rank and args.save:
        # setting up the working directory and recording args
        tag = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        work_dir = os.path.join(
            args.work_dir,
            args.dataset + '-' + args.arch,
            tag
            )
        os.makedirs(work_dir, exist_ok=True)
        print(f"saving experiment product to {work_dir}")
        utils.save_args(args, os.path.join(work_dir, 'args.txt'))
        utils.save_result(result, work_dir)
        # save the trained model
        model_to_save = model if args.distributed is False else \
            model.module
        if FLAGS.quant_mode == "by bit":
            utils.save_model_state(
                model_to_save,
                os.path.join(work_dir, "model_state.pth.tar"))
    print('*' * 40 + "finish" + '*' * 40)


def set_temperature(epoch):
    temp = FLAGS.init_temp * FLAGS.temp_anneal_rate ** \
        (epoch // FLAGS.temp_anneal_step)
    FLAGS.temp = max(temp, FLAGS.temp_min)


def set_randomness(epoch):
    random = FLAGS.init_random * FLAGS.random_anneal_rate ** \
        (epoch // FLAGS.random_anneal_step)
    FLAGS.random = random


def fitting(model, dataset, optimizer, lr_scheduler, epochs,
            device="cuda", verbose=False, distributed=False):
    train_loss, train_acc1, train_acc5 = [], [], []
    val_loss, val_acc1, val_acc5 = [], [], []
    best_acc1 = best_acc5 = 0
    for epoch in range(epochs):
        if distributed is True:
            dataset.train_loader.sampler.set_epoch(epoch)
        print(f"Epoch: {epoch + 1}/{epochs} \t "
              f"lr={lr_scheduler.get_last_lr()[0]}, "
              f"temperature={FLAGS.temp}, random={FLAGS.random}")
        loss, acc1, acc5, _ = train_epoch(
            model, dataset.train_loader, optimizer,
            device=device, verbose=verbose, distributed=distributed)
        # print(model.layer1[0].conv1.alpha[0][0][0])
        train_loss.append(loss)
        train_acc1.append(acc1)
        train_acc5.append(acc5)
        print(f"Training result => Averaged loss: {loss:.5f}, "
              f"acc@1: {acc1:.2%}, acc@5: {acc5:.2%}")
        loss, acc1, acc5, _ = evaluate(
            model, dataset.val_loader,
            device=device, verbose=verbose, distributed=distributed)
        val_loss.append(loss)
        val_acc1.append(acc1)
        val_acc5.append(acc5)
        best_acc1 = max(best_acc1, acc1)
        best_acc5 = max(best_acc5, acc5)
        print(f"Eval result => Averaged loss: {loss:.5f}, acc@1: {acc1:.2%} "
              f"({best_acc1:.2%}), acc@5: {acc5:.2%} "
              f"({best_acc5:.2%})")
        lr_scheduler.step()
        set_randomness(epoch+1)
        set_temperature(epoch+1)
    History = collections.namedtuple("History", [
        "train_loss", "train_acc1", "train_acc5",
        "val_loss", "val_acc1", "val_acc5"])
    history = History(
        train_loss, train_acc1, train_acc5,
        val_loss, val_acc1, val_acc5)
    return history


if __name__ == '__main__':
    main()
