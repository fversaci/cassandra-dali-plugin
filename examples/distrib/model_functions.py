import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import sys
import time
import shutil
import numpy as np
import math
from tqdm import trange, tqdm

import model_initialization as mi

try:
    from apex import amp
except:
    print("Failed importing apex.amp module")

def train(train_loader, model, num_classes, criterion, optimizer, epoch, tot_epochs, local_rank, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()
    
    if local_rank == 0:
        pbar = tqdm(train_loader)
        print ()
        print ("-"*8)
        print ("TRAINING" )
        print ("-"*8)

    else:
        pbar = train_loader
    
    train_loader_len = int(math.ceil(train_loader._size / args.batch_size))

    # switch to train mode
    model.train()
    end = time.time()
    
    for i, data in enumerate(pbar):
        input = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()

        if args.prof >= 0 and i == args.prof:
            print("Profiling begun at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStart()

        if args.prof >= 0:
            torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        adjust_learning_rate(optimizer, epoch, i, train_loader_len, args.lr)
        if args.test:
            if i > 10:
                break

        # compute output
        if args.prof >= 0:
            torch.cuda.nvtx.range_push("forward")
        output = model(input)

        if args.prof >= 0:
            torch.cuda.nvtx.range_pop()
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.prof >= 0:
            torch.cuda.nvtx.range_push("backward")
        
        if args.opt_level is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        if args.prof >= 0:
            torch.cuda.nvtx.range_pop()

        if args.prof >= 0:
            torch.cuda.nvtx.range_push("optimizer.step()")
        
        optimizer.step()
        
        if args.prof >= 0:
            torch.cuda.nvtx.range_pop()

        # Every print_freq iterations, check the loss, accuracy,
        # and speed.  For best performance, it doesn't make sense
        # to print these metrics every iteration, since they incur
        # an allreduce and some host<->device syncs.

        # Measure accuracy
        prec1, preck = accuracy(output.data, target, topk=(1, min(5, num_classes)))

        # Average loss and accuracy across processes for logging
        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            prec1 = reduce_tensor(prec1)
            preck = reduce_tensor(preck)
        else:
            reduced_loss = loss.data

        # to_python_float incurs a host<->device sync
        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        topk.update(to_python_float(preck), input.size(0))

        torch.cuda.synchronize()
        #batch_time.update((time.time() - end) / args.print_freq)
        batch_time.update((time.time() - end))
        end = time.time()

        # Progression bar management
        if local_rank == 0:
            speed_val = args.world_size * args.batch_size / batch_time.val
            speed_avg = args.world_size * args.batch_size / batch_time.avg
            k = min(5,num_classes)
            msg = f"Epoch {epoch}/{tot_epochs} - Time, {batch_time.val:.3f}, {batch_time.avg:.3f} - Speed, {speed_val:.3f}, {speed_avg:.3f} - loss, {losses.val:.3f}, {losses.avg:.3f} - top-1-acc, {top1.val:.2f}, {top1.avg:.2f} - top-{k}-acc, {topk.val:.2f}, {topk.avg:.2f}"
            pbar.set_postfix_str(msg)
        
        # Pop range "Body of iteration {}".format(i)
        if args.prof >= 0:
            torch.cuda.nvtx.range_pop()

        if args.prof >= 0 and i == args.prof + 10:
            print("Profiling ended at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStop()
            quit()
        
    if local_rank == 0:
        pbar.close()
    
    return batch_time.avg


def validate(val_loader, model, num_classes, criterion, local_rank, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    
    if local_rank == 0:
        pbar = tqdm(val_loader)
        print ()
        print ("-"*10)
        print ("VALIDATION" )
        print ("-"*10)

    else:
        pbar = val_loader

    val_loader_len = int(val_loader._size / args.batch_size)

    for i, data in enumerate(pbar):
        input = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)
        
        # measure accuracy and record loss
        prec1, preck = accuracy(output.data, target, topk=(1, min(5,num_classes)))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            preck = reduce_tensor(preck)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        topk.update(to_python_float(preck), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if local_rank == 0:
            speed_val = args.world_size * args.batch_size / batch_time.val
            speed_avg = args.world_size * args.batch_size / batch_time.avg
            k = min(5,num_classes)
            
            msg = f"Time, {batch_time.val:.3f}, {batch_time.avg:.3f} - Speed, {speed_val:.3f}, {speed_avg:.3f} - loss, {losses.val:.3f}, {losses.avg:.3f} - top-1-acc, {top1.val:.2f}, {top1.avg:.2f} - top-{k}-acc, {topk.val:.2f}, {topk.avg:.2f}"
            pbar.set_postfix_str(msg)
    
    if local_rank == 0:
        pbar.close()
        
    print(" * Acc@1 {top1.avg:.2f} Acc@k {topk.avg:.2f}".format(top1=top1, topk=topk))

    return [top1.avg, topk.avg]


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, step, len_epoch, start_lr):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = start_lr * (0.1 ** factor)

    """Warmup"""
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5.0 * len_epoch)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, "item"):
        return t.item()
    else:
        return t[0]

