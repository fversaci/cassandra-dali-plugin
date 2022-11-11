# Adapted to use cassandra-dali-plugin, from
# https://github.com/NVIDIA/DALI/blob/main/docs/examples/use_cases/pytorch/resnet50/main.py
# (Apache License, Version 2.0)
#
# Run with:
# python3 -m torch.distributed.launch --nproc_per_node=NUM_GPUS distrib_train_from_cassandra.py -a resnet50 --dali_cpu --b 128 --loss-scale 128.0 --workers 4 --lr=0.4 --opt-level O2 --keyspace=imagenette --train-table-suffix=train_orig --val-table-suffix=val_orig

# cassandra reader
from cassandra_reader import get_cassandra_reader, get_cassandra_reader_from_splitfile, get_cassandra_row_data

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
import cassandra_feature_writer as cfw

from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

def parse():
    model_names = sorted(
        name
        for name in models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(models.__dict__[name])
    )

    parser = argparse.ArgumentParser(description="PyTorch feature extraction and store")
    parser.add_argument(
        "--keyspace",
        "-k",
        metavar="KEYSP",
        default="imagenette",
        help="cassandra keyspace, i.e., dataset name (default: imagenette)",
    )
    parser.add_argument(
        "--input-table-suffix",
        metavar="SUFF",
        help="Suffix for input table names",
    )
    parser.add_argument(
        "--output-table-suffix",
        metavar="SUFF",
        help="Suffix for output table names",
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
        "--weight-fn",
        "-w",
        metavar="WFN",
        default=None,
        help="filename of the parameter to be load to the network",
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
        "-b",
        "--batch-size",
        default=64,
        type=int,
        metavar="BS",
        help="batch size (default: 64)",
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
        "--dali-cpu",
        action="store_true",
        help="Runs CPU based version of DALI pipeline.",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--channels-last", type=bool, default=False)
    args = parser.parse_args()
    return args


def main():
    global args, local_rank
    local_rank = int(os.environ["LOCAL_RANK"])
    args = parse()

    print (f"Local_rank: {local_rank}")

    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

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

    # create model
    num_classes = args.num_classes
    freeze_params = True
    new_top=True
    pretrained_weights = True
    weight_fn = args.weight_fn
    get_features=True
    model_name = args.arch
    
    print (f"Creating model... {model_name}")
    model = mi.initialize_model(model_name, num_classes, freeze_params=freeze_params, \
            pretrained_weights=pretrained_weights, new_top=new_top, weights_fn=weight_fn,\
            get_features=get_features)

    if hasattr(torch, "channels_last") and hasattr(torch, "contiguous_format"):
        if args.channels_last:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
        model = model.cuda().to(memory_format=memory_format)
    else:
        model = model.cuda()

    if args.distributed:
        model = DDP(model)

    crop_size = 256
    val_size = 256

    print ("Creating DALI Data Pipeline")
    # val pipe
    pipe = cdp.create_dali_pipeline(
        keyspace=args.keyspace,
        table_suffix=args.input_table_suffix,
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
        split_fn=None,
        split_index=0
    )
    pipe.build()
    data_loader = DALIClassificationIterator(
        pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL
    )
    
    ## Get input db metadata
    print ("Reading input metadata")
    in_meta = get_cassandra_row_data(args.keyspace, args.input_table_suffix)

    ## Cassandra writer    
    if args.output_table_suffix:
        cw = cfw.get_cassandra_feature_writer(args.keyspace, args.output_table_suffix)
        output_mode = 'db'
    else:
        cw = None
        output_mode = 'debug'
    
    ### Inference on data set and write out results on db or printing them
    mf.write_features_to_table(data_loader, cw, in_meta, model, local_rank, args, output_mode)

if __name__ == "__main__":
    main()
