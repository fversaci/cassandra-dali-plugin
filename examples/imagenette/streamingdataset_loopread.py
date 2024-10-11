import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from streaming import (
    StreamingDataset,
)  # assuming StreamingDataset is imported from a package like 'streaming'
from streaming.vision import StreamingImageNet
from tqdm import tqdm, trange
from IPython import embed
from clize import run
import numpy as np
import time 

global_rank = int(os.getenv("RANK", default=0))
local_rank = int(os.getenv("LOCAL_RANK", default=0))
world_size = int(os.getenv("WORLD_SIZE", default=1))


# Initialize process for distributed training
def setup(rank, world_size):
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def scan(root_dir="s3://imagenet/streaming/", split="train", bs=1024, epochs=4, log_fn=None, local_cache="/tmp/streamingdata_loopread/"):
    # Set up distributed environment
    rank = local_rank
    setup(rank, world_size)

    #os.environ["MOSAICML_STREAMING_AWS_REQUESTER_PAYS"] = "1"
    # remove previous cache file if any
    try:
        os.rmdir(os.path.join(local_cache, split))
    except:
        print ("local cache does not exist")

    dataset = StreamingDataset(
    remote=root_dir,  # Your remote storage
    local=local_cache, # Local cache
    split=split,                       # Training split
    batch_size=bs,
    shuffle=True,
    shuffle_block_size=bs*4,
    cache_limit=10e9,
    )

    # Create the DataLoader for distributed training
    data_loader = DataLoader(
        dataset,
        batch_size = bs // world_size,
        num_workers=4,
        #pin_memory=True,
    )
    
    steps = len(data_loader)

    ### Start scanning
    print(f"Batches per epoch: {steps}")

    timestamps_np = np.zeros((epochs, steps))
    batch_bytes_np = np.zeros((epochs, steps))

    first_epoch = True

    for epoch in range(epochs):
        # read data for current epoch
        with tqdm(data_loader) as t:
            start_ts = time.time()
            for step, data in enumerate(t):
                images = data['x']
                labels = data['y']
                
                #if step == 0:
                #    embed()
                
                images_batch_bytes = sum([len(dd) for dd in images])
                labels_batch_bytes = labels.shape[0] * 8 # label is longtensor (int 64bit)
                batch_bytes = images_batch_bytes + labels_batch_bytes

                batch_bytes_np[epoch][step] = batch_bytes
                timestamps_np[epoch, step] = time.time() - start_ts
                start_ts = time.time()
                
                
    # Calculate the average and standard deviation
    if epochs > 3:
        # First epoch is skipped
        ## Speed im/s
        average_io_GBs_per_epoch = (
            np.sum(batch_bytes_np[1:], axis=1) / np.sum(timestamps_np[1:], axis=1)
        ) / 1e9
        std_dev_io_GBs = np.std(average_io_GBs_per_epoch)
        average_time_per_epoch = np.mean(timestamps_np[1:], axis=(1))
        std_dev_time = np.std(average_time_per_epoch)
        average_speed_per_epoch = bs / average_time_per_epoch
        std_dev_speed = np.std(average_speed_per_epoch)

        print(f"Stats for epochs > 1, batch_size = {bs}")
        print(
            f"  Average IO: {np.mean(average_io_GBs_per_epoch):.2e} ± {std_dev_io_GBs:.2e} GB/s"
        )
        print(
            f"  Average batch time: {np.mean(average_time_per_epoch):.2e} ± {std_dev_time:.2e} s"
        )
        print(
            f"  Average speed: {np.mean(average_speed_per_epoch):.2e} ± {std_dev_speed:.2e} im/s"
        )

    if log_fn:
        data = (bs, timestamps_np, batch_bytes_np)
        pickle.dump(data, open(log_fn, "wb"))

    if world_size > 1:
        cleanup()

if __name__ == "__main__":
    run(scan)
