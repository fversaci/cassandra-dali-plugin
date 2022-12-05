# Adapted to use cassandra-dali-plugin, from
# https://github.com/NVIDIA/DALI/blob/main/docs/examples/use_cases/pytorch/resnet50/main.py
# (Apache License, Version 2.0)
#
# Run with:

# torchrun --nproc_per_node=1 distrib_train_from_cassandra_splitfile.py -a resnet18 --b 256 --loss-scale 128.0 --workers 4 --lr=1e-4  --split-fn splitfile_pickle_format  --train-split-index 0 --val-split-index 1 --num-classes 2

# torchrun --nproc_per_node=1 distrib_train_from_cassandra_splitfile.py -a resnet18 --b 256 --loss-scale 128.0 --workers 4 --lr=1e-4 --opt-level O2 --keyspace=imagenette --train-table-suffix=train_orig --val-table-suffix=val_orig 

# cassandra reader
import argparse
import os
import shutil
import time
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.models as models

from tqdm import trange, tqdm
import numpy as np

import model_initialization as mi
import model_functions as mf
import cassandra_dali_pipeline as cdp

from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

def parse():
    model_names = sorted(
        name
        for name in models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(models.__dict__[name])
    )

    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument(
        "--keyspace",
        "-k",
        metavar="KEYSP",
        default="imagenette",
        help="cassandra keyspace, i.e., dataset name (default: imagenette)",
    )
    parser.add_argument(
        "--train-table-suffix",
        metavar="SUFF",
        default="train_orig",
        help="Suffix for table names (default: orig)",
    )
    parser.add_argument(
        "--val-table-suffix",
        metavar="SUFF",
        default="val_orig",
        help="Suffix for table names (default: orig)",
    )
    parser.add_argument(
        "--data-col",
        metavar="DATACOL",
        default="data",
        help="name of the table column containing data. Used only if no splitfile is loaded",
    )
    parser.add_argument(
        "--label-col",
        metavar="LABELCOL",
        default="label",
        help="name of the table column containing label info. Used only if no splitfile is loaded",
    )
    parser.add_argument(
        "--split-fn",
        metavar="SPF",
        default=None,
        help="splitfile filename",
    )
    parser.add_argument(
        "--train-split-index",
        default=0,
        type=int,
        help="Specify which split index has to be used for the training set",
    )
    parser.add_argument(
        "--val-split-index",
        default=None,
        type=int,
        help="Specify which split index has to be used for the validation set",
    )
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="resnet18",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
    )
    parser.add_argument(
        "-n",
        "--num-classes",
        default=2,
        type=int,
        metavar="N",
        help="number of output classes (default: 2)",
    )
    parser.add_argument(
        "--crop-size",
        default=224,
        type=int,
        metavar="CSIZE",
        help="size of the crop during training step (default: 224px)",
    )
    parser.add_argument(
        "--val-size",
        default=256,
        type=int,
        metavar="VSIZE",
        help="size of the crop during validation step (default: 224px)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size per process (default: 256)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--print-freq",
        "-p",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="use pre-trained model",
    )

    parser.add_argument(
        "--dali_cpu",
        action="store_true",
        help="Runs CPU based version of DALI pipeline.",
    )
    parser.add_argument(
        "--prof", default=-1, type=int, help="Only run 10 iterations for profiling."
    )
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--sync_bn", action="store_true", help="enabling apex sync BN.")

    parser.add_argument("--opt-level", type=str, default=None)
    parser.add_argument("--keep-batchnorm-fp32", type=str, default=None)
    parser.add_argument("--loss-scale", type=str, default=None)
    parser.add_argument("--channels-last", type=bool, default=False)
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Launch test mode with preset arguments",
    )
    args = parser.parse_args()
    return args


def main():
    global best_prec1, args, local_rank
    best_prec1 = 0
    local_rank = int(os.environ["LOCAL_RANK"])
    args = parse()

    print (f"Local_rank: {local_rank}")

    # test mode, use default args for sanity test
    if args.test:
        args.opt_level = None
        args.epochs = 1
        args.start_epoch = 0
        args.arch = "resnet50"
        args.batch_size = 64
        args.sync_bn = False
        print("Test mode - no DDP, no apex, RN50, 10 iterations")

    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

    # make apex optional
    if args.opt_level is not None or args.distributed or args.sync_bn:
        try:
            global DDP, amp, optimizers, parallel
            from apex.parallel import DistributedDataParallel as DDP
            from apex import amp, optimizers, parallel
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to run this example."
            )

    print("opt_level = {}".format(args.opt_level))
    print(
        "keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32),
        type(args.keep_batchnorm_fp32),
    )
    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    best_prec1 = 0
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(local_rank)
        torch.set_printoptions(precision=10)

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()

    args.total_batch_size = args.world_size * args.batch_size
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # create model
    num_classes = args.num_classes
    freeze_params = False
    new_top=True
    pretrained_weights = True
    checkpoint_fn = None
    get_features=False
    model_name = args.arch
    
    print (f"Creating model... {model_name}")
    model = mi.initialize_model(model_name, num_classes, freeze_params=freeze_params, \
            pretrained_weights=pretrained_weights, new_top=new_top, checkpoint_fn=checkpoint_fn,\
            get_features=get_features)

    if args.sync_bn:
        print("using apex synced BN")
        model = parallel.convert_syncbn_model(model)

    if hasattr(torch, "channels_last") and hasattr(torch, "contiguous_format"):
        if args.channels_last:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
        model = model.cuda().to(memory_format=memory_format)
    else:
        model = model.cuda()

    # Scale learning rate based on global batch size
    args.lr = args.lr * float(args.batch_size * args.world_size) / 256.0
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Initialize Amp.  Amp accepts either values or strings for the
    # optional override arguments, for convenient interoperation with
    # argparse.
    if args.opt_level is not None:
        model, optimizer = amp.initialize(
            model,
            optimizer,
            opt_level=args.opt_level,
            keep_batchnorm_fp32=args.keep_batchnorm_fp32,
            loss_scale=args.loss_scale,
        )

    # For distributed training, wrap the model with
    # apex.parallel.DistributedDataParallel.  This must be done AFTER
    # the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to
    # amp.initialize may alter the types of model's parameters in a
    # way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps
        # communication with computation in the backward pass.  model
        # = DDP(model) delay_allreduce delays all communication to the
        # end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(
                    args.resume,
                    map_location=lambda storage, loc: storage.cuda(args.gpu),
                )
                args.start_epoch = checkpoint["epoch"]
                global best_prec1
                best_prec1 = checkpoint["best_prec1"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                print(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        args.resume, checkpoint["epoch"]
                    )
                )
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        resume()

    if args.arch == "inception_v3":
        raise RuntimeError("Currently, inception_v3 is not supported by this example.")
        # crop_size = 299
        # val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = args.crop_size
        val_size = args.val_size

    print ("Creating DALI Training Pipeline")
    # train pipe
    pipe = cdp.create_dali_pipeline(
        keyspace=args.keyspace,
        table_suffix=args.train_table_suffix,
        batch_size=args.batch_size,
        num_threads=args.workers,
        device_id=local_rank,
        seed=12 + local_rank,
        crop=crop_size,
        size=val_size,
        dali_cpu=args.dali_cpu,
        shard_id=local_rank,
        num_shards=args.world_size,
        is_training=True,
        split_fn=args.split_fn,
        split_index=args.train_split_index,
        data_col=args.data_col,
        label_col=args.label_col
    )
    pipe.build()
    train_loader = DALIClassificationIterator(
        pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL
    )

    print ("Creating DALI Validation Pipeline")
    # val pipe
    pipe = cdp.create_dali_pipeline(
        keyspace=args.keyspace,
        table_suffix=args.val_table_suffix,
        batch_size=args.batch_size,
        num_threads=args.workers,
        device_id=local_rank,
        seed=12 + local_rank,
        crop=crop_size,
        size=val_size,
        dali_cpu=args.dali_cpu,
        shard_id=local_rank,
        num_shards=args.world_size,
        is_training=False,
        split_fn=args.split_fn,
        split_index=args.val_split_index,
        data_col=args.data_col,
        label_col=args.label_col
    )
    pipe.build()
    val_loader = DALIClassificationIterator(
        pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL
    )

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    total_time = mf.AverageMeter()

    ### Training and validation of the model
    ### Looping across epochs
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        avg_train_time = mf.train(train_loader, model, num_classes, criterion, optimizer, epoch, args.epochs, local_rank=local_rank, args=args)
        total_time.update(avg_train_time)
        if args.test:
            break

        # evaluate on validation set
        [prec1, preck] = mf.validate(val_loader, model, num_classes, criterion, local_rank=local_rank, args=args)

        # remember best prec@1 and save checkpoint
        if local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            mf.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
            )
            if epoch == args.epochs - 1:
                print(
                    "##Top-1 {0}\n"
                    "##Top-5 {1}\n"
                    "##Perf  {2}".format(
                        prec1, preck, args.total_batch_size / total_time.avg
                    )
                )

        train_loader.reset()
        val_loader.reset()

if __name__ == "__main__":
    main()
