# Adapted to use cassandra-dali-plugin, from
# https://github.com/NVIDIA/DALI/blob/main/docs/examples/use_cases/pytorch/resnet50/main.py
# (Apache License, Version 2.0)

# cassandra reader
from cassandra_reader import get_cassandra_reader, read_uuids
from create_dali_pipeline import create_dali_pipeline

import argparse

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

import lightning as L
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import CSVLogger

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
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
        choices=["train_orig", "train_256_jpg", "train_512_jpg"],
        help="Suffix for table names (default: orig)",
    )
    parser.add_argument(
        "--val-table-suffix",
        metavar="SUFF",
        default="val_orig",
        choices=["val_orig", "val_256_jpg", "val_512_jpg"],
        help="Suffix for table names (default: orig)",
    )
    parser.add_argument(
        "--ids-cache-dir",
        metavar="CACH",
        default="ids_cache",
        help="Directory containing the cached list of UUIDs (default: ./ids_cache)",
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
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
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
        help="Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.",
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
        "--weights",
        dest="weights",
        type=str,
        metavar="WEIGHTS",
        default=None,
        help="specifies weights to be used. Default: None",
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
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--sync_bn", action="store_true", help="enabling apex sync BN.")

    parser.add_argument("--opt-level", type=str, default=None)
    parser.add_argument("--keep-batchnorm-fp32", type=str, default=None)
    parser.add_argument("--loss-scale", type=str, default=None)
    parser.add_argument("--channels-last", type=bool, default=False)
    
    args = parser.parse_args()
    return args


########################
### LIGHTNING MODELS ###
########################

## Base class: no DALI 

class ImageNetLightningModel(L.LightningModule):
    MODEL_NAMES = sorted(
        name
        for name in models.__dict__
        if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
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

        print('*' * 80)
        print(f'*************** Loading model {self.arch}')
        print('*' * 80)
        self.model = models.__dict__[self.arch](weights=self.weights)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch[0]["data"]
        target = batch[0]["label"].squeeze(-1).long()
        
        # Get output from the model
        output = self(images)

        #print (f"output_shape: {output.shape}, target_shape: {target.shape}")

        loss_train = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log("train_loss", loss_train, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_acc1", acc1, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_acc5", acc5, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return loss_train

    def eval_step(self, batch, batch_idx, prefix: str):
        images = batch[0]["data"]
        target = batch[0]["label"].squeeze(-1).long()
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log(f"{prefix}_loss", loss_val, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, logger=True)
        self.log(f"{prefix}_acc1", acc1,  prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, logger=True)
        self.log(f"{prefix}_acc5", acc5, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True, logger=True)

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
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30))
        return [optimizer], [scheduler]


## Derived class to add the Cassandra DALI pipeline 

class DALI_ImageNetLightningModel(ImageNetLightningModel):
    def __init__(self, 
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
        train_pipeline = self.GetPipeline(args, args.train_table_suffix, is_training=True, device_id=device_id, shard_id=shard_id, num_shards=num_shards, num_threads=8)
        val_pipeline = self.GetPipeline(args, args.val_table_suffix, is_training=False, device_id=device_id, shard_id=shard_id, num_shards=num_shards, num_threads=8)

        # Wrapper class to allow for adding code to the methods
        class LightningWrapper(DALIClassificationIterator):
            def __init__(self, *kargs, **kvargs):
                super().__init__(*kargs, **kvargs)

            def __next__(self):
                out = super().__next__()
                return out

        # Creatind actual loaders used by the lightning module to get data (train_dataloader and val_dataloader methods)
        self.train_loader = LightningWrapper(train_pipeline, last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True, reader_name="Reader")
        self.val_loader = LightningWrapper(val_pipeline, last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True, reader_name="Reader")
    
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
   
    # def on_train_epoch_end(self):
    #     self.train_loader.reset()
    # 
    # def on_validation_epoch_end(self):
    #     self.val_loader.reset()

    @staticmethod
    def GetPipeline(args, table_suffix, is_training, device_id, shard_id, num_shards, num_threads):
        in_uuids = read_uuids(
            keyspace = args.keyspace,
            table_suffix = table_suffix,
            ids_cache_dir = args.ids_cache_dir,
        )
        
        pipe = create_dali_pipeline(
            keyspace = args.keyspace,
            table_suffix = table_suffix,
            batch_size = args.batch_size,
            bs = args.batch_size,
            num_threads = args.workers,
            shard_id = shard_id,
            num_shards = num_shards,
            source_uuids = in_uuids,
            device_id = device_id,
            seed = 1234,  # must be a fixed number for all the ranks to have the same reshuffle across epochs and ranks
            crop = args.crop_size,
            size = args.val_size,
            dali_cpu = args.dali_cpu,
            is_training = is_training,
        )
        pipe.build()
        
        return pipe


def main():
    global best_prec1, args
    best_prec1 = 0
    args = parse()

    if args.profile:
        profiler = PyTorchProfiler(filename="perf-logs")
    else:
        profiler = None

    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    
    if args.arch == "inception_v3":
        raise RuntimeError("Currently, inception_v3 is not supported by this example.")
        # crop_size = 299
        # val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        args.crop_size = 224
        args.val_size = 256


    # create lightning model
    model = DALI_ImageNetLightningModel(args)

    # Optionally resume from a checkpoint
    if args.resume:
        pass # FIXME: TBD
        """
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
        """

    #if args.evaluate:
    #    validate(val_loader, model, criterion)
    #    return

    tensorboard_logger = TensorBoardLogger("tb_logs", name="my_model")
    csv_logger = logger = CSVLogger("logs_csv", name="my_model")
    logger = csv_logger

    ### Lightning Trainer
    if args.num_gpu > 1:
        trainer = L.Trainer(max_epochs=args.epochs,  accelerator="gpu", devices=args.num_gpu, strategy='ddp', profiler=profiler, num_sanity_val_steps=0, logger=logger)
    else:
        trainer = L.Trainer(max_epochs=args.epochs,  accelerator="gpu", devices=1, profiler=profiler, num_sanity_val_steps=0, logger=logger)
    
    trainer.fit(model)


if __name__ == "__main__":
    main()
