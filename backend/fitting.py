import torch
import torch.nn.functional as F
import time
# from .utils import synchronize_all


class ProgressMeter():
    def __init__(self, total, metrics, format, bar_width=30):
        self.bar_width = bar_width
        self.total = total
        self.step = 0
        self.elapsed_time, self.tic = 0, time.time()
        self.stats = {m: 0 for m in metrics}
        self.format = format

    def update(self, *stats):
        if self.step == self.total:
            self._reset()
            return
        self.step += 1
        toc = time.time()
        self.elapsed_time += toc - self.tic
        self.tic = toc
        for i, k in enumerate(self.stats):
            self.stats[k] = stats[i]

    def display(self):
        progress = int(self.bar_width * self.step / self.total)
        bar = '|' + '=' * progress + '>' * (self.bar_width - progress) + '|'
        stats = f"{self.step / self.total:.2%}-{self.elapsed_time:.1f}s "
        for fmt, (k, v) in zip(self.format, self.stats.items()):
            stats += (k + ": {" + fmt + "} ").format(v)
        print('\r'+bar+stats, end='')
        if progress == self.bar_width:
            print()

    def _reset(self):
        self.step = 0
        self.elapsed_time, self.tic = 0, time.time()
        for k in self.stats.keys():
            self.stats[k] = 0


def _loss(output, target):
    # print(output.size(), target.size())
    return F.cross_entropy(output, target)


def batch_forward(model, input_batch, label_batch, loss_func=None):
    output_batch = model(input_batch)
    loss_func = loss_func or _loss
    loss = loss_func(output_batch, label_batch)
    correction1 = count_correction(output_batch, label_batch, 1)
    correction5 = count_correction(output_batch, label_batch, 5)
    return loss, correction1, correction5


def batch_optimize(model, optimizer, input_batch, label_batch):
    loss, correction1, correction5 = batch_forward(
        model, input_batch, label_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, correction1, correction5


def batch_evaluate(model, input_batch, label_batch):
    with torch.no_grad():
        loss, correction1, correction5 = batch_forward(
            model, input_batch, label_batch)
    return loss, correction1, correction5


def train_epoch(model, data_loader, optimizer, device="cuda", verbose=False):
    model.train()
    running_loss = running_total = 0
    running_correct1 = running_correct5 = 0
    # start = time.time()
    meter = ProgressMeter(
        len(data_loader),
        ('loss', 'acc@1', 'acc@5'),
        (":.5f", ":.2%", ":.2%")
        )
    for i, (input_batch, label_batch) in enumerate(data_loader):
        input_batch = input_batch.to(device, non_blocking=True)
        label_batch = label_batch.to(device, non_blocking=True)
        loss, correction1, correction5 = batch_optimize(
            model, optimizer,
            input_batch, label_batch)

        running_loss += loss.item() * input_batch.size(0)
        running_correct1 += correction1.item()
        running_correct5 += correction5.item()
        running_total += input_batch.size(0)
        avg_loss = running_loss / running_total
        avg_acc1 = running_correct1 / running_total
        avg_acc5 = running_correct5 / running_total
        meter.update(avg_loss, avg_acc1, avg_acc5)
        if verbose is True:
            meter.display()
    # print(meter.elapsed_time)
    return avg_loss, avg_acc1, avg_acc5, meter.elapsed_time


def evaluate(model, data_loader, device="cuda", verbose=False):
    model.eval()
    running_loss = running_total = 0
    running_correct1 = running_correct5 = 0
    meter = ProgressMeter(
        len(data_loader),
        ('loss', 'acc@1', 'acc@5'),
        (":.5f", ":.2%", ":.2%")
        )
    for i, (input_batch, label_batch) in enumerate(data_loader):
        input_batch = input_batch.to(device, non_blocking=True)
        label_batch = label_batch.to(device, non_blocking=True)
        loss, correction1, correction5 = batch_evaluate(
            model, input_batch, label_batch)

        running_loss += loss.item() * input_batch.size(0)
        running_correct1 += correction1.item()
        running_correct5 += correction5.item()
        running_total += input_batch.size(0)
        avg_loss = running_loss / running_total
        avg_acc1 = running_correct1 / running_total
        avg_acc5 = running_correct5 / running_total
        if verbose is True:
            meter.update(avg_loss, avg_acc1, avg_acc5)
            meter.display()
    return avg_loss, avg_acc1, avg_acc5, meter.elapsed_time


def count_correction(output_batch, label_batch, k=1):
    _, prediction = output_batch.topk(k)
    label_batch = label_batch.unsqueeze(-1).expand_as(prediction)
    correction = (prediction == label_batch).sum()
    return correction
