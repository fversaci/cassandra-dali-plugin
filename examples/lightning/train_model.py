# Adapted to use cassandra-dali-plugin, from
# https://github.com/NVIDIA/DALI/blob/main/docs/examples/use_cases/pytorch/resnet50/main.py
# (Apache License, Version 2.0)

# cassandra reader
from cassandra_reader import get_cassandra_reader, read_uuids
from create_dali_pipeline import (
    create_dali_pipeline_from_file,
    create_dali_pipeline_cassandra,
    create_dali_pipeline_from_tfrecord,
)

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

try:
    from nvidia.dali.plugin.pytorch import (
        DALIClassificationIterator,
        DALIGenericIterator,
        LastBatchPolicy,
    )
except ImportError:
    raise ImportError(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example."
    )


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
        "--train-folder",
        metavar="DIRECTORY (training)",
        default="",
        help="training folder with a subfolder for each class. This override cassandra and tfrecord",
    )
    parser.add_argument(
        "--val-folder",
        metavar="DIRECTORY (validation)",
        default="",
        help="validation folder with a subfolder for each class. This override cassandra and tfrecord",
    )
    parser.add_argument(
        "--train-tfr-folder",
        metavar="TFRECORD DIRECTORY (training)",
        default="",
        help="training folder with tfrecords. This override cassandra",
    )
    parser.add_argument(
        "--val-tfr-folder",
        metavar="TFRECORD DIRECTORY (validation)",
        default="",
        help="validation folder with tfrecords. This override cassandra",
    )
    parser.add_argument(
        "--train-index-folder",
        metavar="TFRECORD INDEX DIRECTORY (training)",
        default="",
        help="training folder with tfrecord index. This override cassandra",
    )
    parser.add_argument(
        "--val-index-folder",
        metavar="TFRECORD INDEX DIRECTORY (validation)",
        default="",
        help="validation folder with tfrecord index. This override cassandra",
    )
    parser.add_argument(
        "--train-data-table",
        metavar="DATA TABLE (training)",
        default="imagenette.data_train",
        help="cassandra training data table (i.e.: keyspace.tablename (default: imagenette.data_train)",
    )
    parser.add_argument(
        "--train-rows-fn",
        metavar="Local copy of UUIDs (training)",
        default="train.rows",
        help="Local copy of training UUIDs (default: train.rows)",
    )
    parser.add_argument(
        "--val-data-table",
        metavar="DATA TABLE (validation)",
        default="imagenette.data_val",
        help="cassandra validation data table (i.e.: keyspace.tablename (default: imagenette.data_val)",
    )
    parser.add_argument(
        "--val-rows-fn",
        metavar="Local copy of UUIDs (validation)",
        default="val.rows",
        help="Local copy of training UUIDs (default: val.rows)",
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


#######################
### TENSOR ITERATOR ###
#######################

## This iterator is used to feed GPU with the same tensor without loading data
## It is used to evaluate the upper bound performance of the GPU training a specific model
## It is activated by the command line argument --no-io


class TensorIterator:
    def __init__(self, model, size=(3, 224, 224), batch_size=256, data_size=10000):
        self.size = size
        self.bs = batch_size
        self.model = model
        self.data_size = data_size
        self.cnt = 0

        t = torch.rand(
            self.bs, self.size[0], self.size[1], self.size[2], device=self.model.device
        )
        fake_label = torch.randint(0, 999, (self.bs,), device=self.model.device)
        self.batch = [{"data": t, "label": fake_label}]

    def __len__(self):
        return self.data_size // self.bs

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt == self.data_size // self.bs:
            self.cnt = 0
            raise StopIteration
        else:
            self.cnt += 1
            return self.batch

    def reset(self):
        self.cnt = 0


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
        images = batch[0]["data"]
        target = batch[0]["label"].squeeze(-1).long()

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
        images = batch[0]["data"]
        target = batch[0]["label"].squeeze(-1).long()

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


############################################################################################################################
## Derived class to use either DALI (file reader, tfrecord reader, Cassandra plugin reader) or the no-io Lightning module ##
############################################################################################################################


class DALI_ImageNetLightningModel(ImageNetLightningModel):
    def __init__(
        self,
        args,
    ):
        super().__init__(**vars(args))

    def prepare_data(self):
        # no preparation is needed in DALI
        # All the preprocessing steps are performed within the DALI pipeline
        pass

    def setup(self, stage=None):
        ## Get info for distributed setup
        device_id = self.local_rank
        shard_id = self.global_rank
        num_shards = self.trainer.world_size

        # Create DALI pipelines (with cassandra plugins)
        train_pipeline = self.GetPipeline(
            args,
            is_training=True,
            device_id=device_id,
            shard_id=shard_id,
            num_shards=num_shards,
        )
        val_pipeline = self.GetPipeline(
            args,
            is_training=False,
            device_id=device_id,
            shard_id=shard_id,
            num_shards=num_shards,
        )

        # Wrapper class to allow for adding code to the methods
        class LightningWrapper(DALIClassificationIterator):
            def __init__(self, *kargs, **kvargs):
                super().__init__(*kargs, **kvargs)

            def __next__(self):
                out = super().__next__()
                return out

        # Creatind actual loaders used by the lightning module to get data (train_dataloader and val_dataloader methods)
        self.train_loader = LightningWrapper(
            train_pipeline,
            # last_batch_policy=LastBatchPolicy.PARTIAL,
            last_batch_policy=LastBatchPolicy.DROP,
            auto_reset=True,
            reader_name="Reader",
        )
        self.val_loader = LightningWrapper(
            val_pipeline,
            # last_batch_policy=LastBatchPolicy.PARTIAL,
            last_batch_policy=LastBatchPolicy.DROP,
            auto_reset=True,
            reader_name="Reader",
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def on_train_epoch_end(self):
        self.train_loader.reset()

    def on_validation_epoch_end(self):
        self.val_loader.reset()

    @staticmethod
    def GetPipeline(
        args,
        is_training,
        device_id,
        shard_id,
        num_shards,
    ):

        if args.train_folder:
            # Data come from files
            if is_training:
                folder = args.train_folder
            else:
                folder = args.val_folder

            pipe = create_dali_pipeline_from_file(
                batch_size=args.batch_size,
                crop=args.crop_size,
                dali_cpu=args.dali_cpu,
                data_dir=folder,
                device_id=device_id,
                is_training=is_training,
                num_shards=num_shards,
                num_threads=args.workers,
                prefetch_queue_depth=2,
                seed=1234,  # must be a fixed number for all the ranks to have the same reshuffle across epochs and ranks
                shard_id=shard_id,
                size=args.val_size,
            )

        elif args.train_tfr_folder:
            # Data from TFRecord
            if is_training:
                folder = args.train_tfr_folder
                index_folder = args.train_index_folder
            else:
                folder = args.val_tfr_folder
                index_folder = args.val_index_folder

            pipe = create_dali_pipeline_from_tfrecord(
                batch_size=args.batch_size,
                crop=args.crop_size,
                dali_cpu=args.dali_cpu,
                file_root=folder,
                index_root=index_folder,
                device_id=device_id,
                is_training=is_training,
                num_shards=num_shards,
                num_threads=args.workers,
                prefetch_queue_depth=2,
                seed=1234,  # must be a fixed number for all the ranks to have the same reshuffle across epochs and ranks
                shard_id=shard_id,
                size=args.val_size,
            )

        else:
            # Data come from cassandra
            if is_training:
                data_table = args.train_data_table
                rows_fn = args.train_rows_fn
            else:
                data_table = args.val_data_table
                rows_fn = args.val_rows_fn

            in_uuids = read_uuids(rows_fn=rows_fn)

            pipe = create_dali_pipeline_cassandra(
                batch_size=args.batch_size,
                crop=args.crop_size,
                dali_cpu=args.dali_cpu,
                data_table=data_table,
                device_id=device_id,
                is_training=is_training,
                num_shards=num_shards,
                num_threads=args.workers,
                prefetch_queue_depth=2,
                seed=1234,  # must be a fixed number for all the ranks to have the same reshuffle across epochs and ranks
                shard_id=shard_id,
                shuffle_every_epoch=True,
                size=args.val_size,
                source_uuids=in_uuids,
                ooo=args.ooo,
                slow_start=args.slow_start,
                io_threads=args.n_io_threads,
                prefetch_buffers=args.n_prefetch_buffers,
            )

        pipe.build()

        return pipe


### model with no dataloader
class NoIO_ImageNetLightningModel(ImageNetLightningModel):
    def __init__(
        self,
        args,
    ):
        super().__init__(**vars(args))

    def setup(self, stage=None):
        self.data_size = 131072 // self.trainer.world_size

    def prepare_data(self):
        # no preparation is needed in DALI
        # All the preprocessing steps are performed within the DALI pipeline
        pass

    def train_dataloader(self):
        return TensorIterator(
            model=self, batch_size=self.batch_size, data_size=self.data_size
        )

    def val_dataloader(self):
        return TensorIterator(
            model=self, batch_size=self.batch_size, data_size=self.data_size
        )

    def on_train_epoch_end(self):
        self.trainer.train_dataloader.reset()

    def on_validation_epoch_end(self):
        self.trainer.val_dataloaders.reset()


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
    if args.no_io:
        model = NoIO_ImageNetLightningModel(args)
    else:
        model = DALI_ImageNetLightningModel(args)

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
