# Adapted to use cassandra-dali-plugin, from
# https://github.com/NVIDIA/DALI/blob/main/docs/examples/use_cases/pytorch/resnet50/main.py
# (Apache License, Version 2.0)

import os
import argparse
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from streaming.vision import StreamingImageNet
import streaming

import lightning as L
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

import time
import numpy as np

from IPython import embed

from s3_utils import list_s3_files


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
        "--streaming-remote",
        metavar="S3 ADDRESS",
        default=None,
        help="S3 remote address of the streaming dataset",
    )
    parser.add_argument(
        "--streaming-local",
        metavar="PATH",
        default="/tmp/streamingdata_loopread/",
        help="Streaming local cache patch",
    )
    parser.add_argument(
        "--streaming-local-size",
        metavar="SIZE (bytes)",
        default=20e9,
        help="StreamingData local cache size (default:20GB)",
    )
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="resnet50",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of DALI data loading workers (default: 4)",
    )
    parser.add_argument(
        "-g",
        "--num-gpu",
        default=1,
        type=int,
        metavar="N_GPU",
        help="number of gpu to be used (default: 1)",
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
        help="Learning rate.  Will be scaled by the number of workers",
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
        "--patience",
        "-p",
        default=10,
        type=int,
        metavar="PATIENCE_EPOCHS",
        help="number of epochs without validation loss improvment that enalbles early stopping (default: 10)",
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
        "--no-checkpoints",
        dest="no_checkpoints",
        action="store_true",
        help="Disable storing the best and last epoch model as checkpoint",
    )
    parser.add_argument(
        "--log-tensorboard",
        dest="log_tensorboard_fname",
        type=str,
        default="",
        help="Log metrics to tensorboard format",
    )
    parser.add_argument(
        "--log-csv",
        dest="log_csv_fname",
        type=str,
        default="",
        help="Log metrics to csv file",
    )
    parser.add_argument(
        "--pretrained-weights",
        dest="weights",
        type=str,
        metavar="WEIGHTS",
        default=None,
        help="model pretrained weights. Default: None",
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        action="store_true",
        help="profile fit operations",
    )
    parser.add_argument(
        "--dali_cpu",
        action="store_true",
        help="Runs CPU based version of DALI pipeline.",
    )
    parser.add_argument(
        "--prof", default=-1, type=int, help="Only run 10 iterations for profiling."
    )
    parser.add_argument(
        "--no-io",
        dest="no_io",
        action="store_true",
        help="Train model by using a single tensor. No data loading is performed. It is used to evaluate the upper bound of the GPU performance",
    )
    parser.add_argument(
        "--out-of-order",
        dest="ooo",
        action="store_true",
        help="Enable out of order Cassandra data loading",
    )
    parser.add_argument(
        "--slow-start",
        default=0,
        type=int,
        metavar="INT",
        help="Incremental prefetching factor (default: 0)",
    )
    parser.add_argument(
        "--n-io-threads",
        default=4,
        type=int,
        metavar="INT",
        help="Number of the Cassandra plugin IO threads (default:4)",
    )
    parser.add_argument(
        "--n-prefetch-buffers",
        default=2,
        type=int,
        metavar="INT",
        help="Number of the Cassandra plugin prefetch buffers (default: 2)",
    )
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--sync_bn", action="store_true", help="enabling apex sync BN.")

    parser.add_argument("--keep-batchnorm-fp32", type=str, default=None)
    parser.add_argument("--channels-last", type=bool, default=False)

    args = parser.parse_args()
    return args


##################################
### CALLBACKS TO LOG TIMESTAMP ###
##################################


class TrainBatchStartCallback(L.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Record the current timestamp
        batch_ts = time.time()

        # Log the timestamp to the logger
        pl_module.log(
            "train_batch_ts",
            torch.tensor(batch_ts, dtype=torch.double),
            prog_bar=False,  # Don't show in the progress bar
            on_step=True,  # Log it at the step level (every batch)
            on_epoch=False,  # Don't log it at the epoch level
            logger=True,  # Send the log to the logger (e.g., TensorBoard)
            sync_dist=False,  # No need to sync across distributed systems (if not using multiple GPUs)
        )


class ValidationBatchStartCallback(L.Callback):
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Record the current timestamp
        batch_ts = time.time()

        # Log the timestamp to the logger
        pl_module.log(
            "val_batch_ts",
            torch.tensor(batch_ts, dtype=torch.double),
            prog_bar=False,  # Don't show in the progress bar
            on_step=True,  # Log it at the step level (every batch)
            on_epoch=False,  # Don't log it at the epoch level
            logger=True,  # Send the log to the logger (e.g., TensorBoard)
            sync_dist=False,  # No need to sync across distributed systems (if not using multiple GPUs)
        )


########################
### LIGHTNING MODELS ###
########################

## Base class: no DALI


class ImageNetLightningModel(L.LightningModule):
    MODEL_NAMES = sorted(
        name
        for name in models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(models.__dict__[name])
    )

    ## LightningModule has two methods to return global_rank and local_rank taking them from self.trainer.

    def __init__(
        self,
        arch: str = "resnet_50",
        weights: str = None,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        batch_size: int = 4,
        workers: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.arch = arch
        self.weights = weights
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.workers = workers

        print("*" * 80)
        print(f"*************** Loading model {self.arch}")
        print("*" * 80)

        self.model = models.__dict__[self.arch](weights=self.weights)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch[0]
        target = batch[1]

        # Get output from the model
        output = self(images)

        loss_train = F.cross_entropy(output, target)

        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))

        self.log(
            "train_loss",
            loss_train,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "train_acc1",
            acc1,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "train_acc5",
            acc5,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        return loss_train

    def eval_step(self, batch, batch_idx, prefix: str):
        images = batch[0]
        target = batch[1]

        output = self(images)

        loss_val = F.cross_entropy(output, target)

        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))

        self.log(
            f"{prefix}_loss",
            loss_val,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            logger=True,
        )

        self.log(
            f"{prefix}_acc1",
            acc1,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            logger=True,
        )

        self.log(
            f"{prefix}_acc5",
            acc5,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            logger=True,
        )

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k."""
        with torch.no_grad():
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

    def configure_optimizers(self):
        # Scale learning rate based on global batch size.
        # The Actual batch size in a distributed training setup is muliplied by the number of workers k
        # Using the linear Scaling Rule (When scaling the batch size by k, scale the learning rate also by k)
        self.lr = self.lr * self.trainer.world_size
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        # scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30))
        # return [optimizer], [scheduler]
        return [optimizer]


### StreamingDataset
class Streaming_ImageNetLightningModel(ImageNetLightningModel):
    def __init__(
        self,
        args,
    ):
        super().__init__(**vars(args))
        self.rank_set = False

    def prepare_data(self):
        # no preparation is needed in DALI
        # All the preprocessing steps are performed within the DALI pipeline
        pass

    def setup(self, stage=None):
        ## Set environment for distributed execution
        if not self.rank_set:
            world_size = self.trainer.num_devices
            local_rank = self.trainer.strategy.local_rank

            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
            os.environ["LOCAL_RANK"] = str(local_rank)
            os.environ["RANK"] = str(local_rank)

            self.rank_set = True

        ## Get info for distributed setup
        device_id = self.local_rank
        shard_id = self.global_rank
        num_shards = self.trainer.world_size

        bs = args.batch_size
        remote_s3 = args.streaming_remote
        local_cache = args.streaming_local
        predownload_batches = 8
        cache_limit = args.streaming_local_size
        shuffle_batches = 16
        shuffle_block_size = bs * shuffle_batches

        # Transformations
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.val_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(int(3 / x.shape[0]), 1, 1)),
                normalize,
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.Resize(args.val_size),
                transforms.CenterCrop(args.crop_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(int(3 / x.shape[0]), 1, 1)),
                normalize,
            ]
        )
        # remove previous cache file if any

        try:
            streaming.base.util.clean_stale_shared_memory()
            if device_id == 0 and remote_s3 and os.path.exists(local_cache):
                flag = input (f"Local path {local_cache} exists and should be deleted. Do you want to delete it (y/N)?")
                if flag == 'y':
                    shutil.rmtree(local_cache)
        except:
            print("local cache does not exist")

        # Create Streaming Datasets for training and validation
        train_dataset = StreamingImageNet(
            remote=remote_s3,
            local=local_cache,
            split="train",
            predownload=predownload_batches * bs,
            cache_limit=cache_limit,
            batch_size=bs,
            shuffle=True,
            shuffle_algo="py1e",
            shuffle_seed=9176,
            shuffle_block_size=shuffle_block_size,
            transform=train_transform,
        )

        val_dataset = StreamingImageNet(
            remote=remote_s3,
            local=local_cache,
            split="val",
            predownload=predownload_batches * bs,
            cache_limit=cache_limit,
            batch_size=bs,
            shuffle=True,
            shuffle_algo="py1e",
            shuffle_seed=9176,
            shuffle_block_size=shuffle_block_size,
            transform=val_transform,
        )

        # Creatind actual loaders used by the lightning module to get data (train_dataloader and val_dataloader methods)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=bs,
            num_workers=8,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=bs,
            num_workers=8,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass


def main():
    global best_prec1, args
    best_prec1 = 0
    args = parse()

    if args.profile:
        profiler = PyTorchProfiler(filename="perf-logs")
    else:
        profiler = None

    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    if args.arch == "inception_v3":
        raise RuntimeError("Currently, inception_v3 is not supported by this example.")
        # crop_size = 299
        # val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        args.crop_size = 224
        args.val_size = 256

    # create lightning model
    model = Streaming_ImageNetLightningModel(args)

    # Optionally resume from a checkpoint
    if args.resume:
        ckpt_path = args.resume
    else:
        ckpt_path = None

    # if args.evaluate:
    #    validate(val_loader, model, criterion)
    #    return

    ### Callbacks
    train_start_callback = TrainBatchStartCallback()
    val_start_callback = ValidationBatchStartCallback()

    callbacks_l = [train_start_callback, val_start_callback]
    # callbacks_l = []

    if args.patience:
        print(f"-- Early stopping enabled with patience={args.patience}")
        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.patience,
            check_finite=False,
        )
        callbacks_l.append(early_stopping)

    if not args.no_checkpoints:
        print("-- Enabling checkpointing for last epoch and best model")
        checkpoint = ModelCheckpoint(
            monitor="val_acc1",
            mode="max",
            dirpath="checkpoints",
            save_last=True,
            save_top_k=1,
        )
        callbacks_l.append(checkpoint)

    ### Loggers
    loggers_l = []

    if args.log_tensorboard_fname:
        tensorboard_logger = TensorBoardLogger("tb_logs", name=log_tensorboard_fname)
        loggers_l.append(tensorboard_logger)

    if args.log_csv_fname:
        csv_logger = logger = CSVLogger("logs_csv", name=args.log_csv_fname)
        loggers_l.append(csv_logger)

    if not loggers_l:
        loggers_l = None  # To prevent warnings in the case metrics are logged only to the progress bar

    ### Lightning Trainer
    if args.num_gpu > 1:
        trainer = L.Trainer(
            max_epochs=args.epochs,
            accelerator="gpu",
            devices=args.num_gpu,
            strategy="ddp",
            profiler=profiler,
            enable_checkpointing=True,
            logger=loggers_l,
            callbacks=callbacks_l,
            num_sanity_val_steps=0,
            precision="16-mixed",
            log_every_n_steps=1,
        )
    else:
        trainer = L.Trainer(
            max_epochs=args.epochs,
            accelerator="gpu",
            devices=1,
            profiler=profiler,
            enable_checkpointing=True,
            logger=loggers_l,
            callbacks=callbacks_l,
            num_sanity_val_steps=0,
            precision="16-mixed",
            log_every_n_steps=1,
        )

    trainer.fit(model, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
