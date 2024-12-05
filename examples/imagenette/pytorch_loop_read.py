import os, sys
import statistics
from clize import run
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm, trange

from torchdata.datapipes.iter import IterableWrapper
from IPython import embed

import boto3
import time
import numpy as np
import pickle


def parse_s3_uri(s3_uri):
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI")

    # Remove the "s3://" prefix
    s3_uri = s3_uri[5:]

    # Split the remaining part into bucket and prefix
    parts = s3_uri.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    return bucket_name, prefix


def list_s3_files(s3_uri):
    bucket_name, prefix = parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    paths = []
    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                paths.append(f"s3://{bucket_name}/{obj['Key']}")

    return sorted(paths)


class S3Dataset(Dataset):
    def __init__(self, s3_root_dir, transform=None):
        self.s3 = boto3.client('s3')
        self.s3_bucket, self.prefix = parse_s3_uri(s3_root_dir)
        self.file_keys = self.get_file_list(s3_root_dir) # List of keys (file names/paths) in the S3 bucket
        self.transform = transform

    def __len__(self):
        return len(self.file_keys)

    def __getitem__(self, idx):
        fname = self.file_keys[idx][0]
        label = self.file_keys[idx][1]
        
        # Read the file from S3
        response = self.s3.get_object(Bucket=self.s3_bucket, Key=fname)
        
        image = response['Body'].read()

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def get_file_list(self, root_dir): 
        file_keys = list_s3_files(root_dir)
        
        label_index = 0
        label_map = {}

        s3_files = []
        
        for file in file_keys:
            els = file.split('/')[-2:]
            label_name = els[0]
            basename = els[1]

            if label_name not in label_map:
                label_map[label_name] = label_index
                s3_files.append((os.path.join(self.prefix, '/'.join([label_name, basename])), label_index))
                label_index += 1
            
            else:
                s3_files.append((os.path.join(self.prefix, '/'.join([label_name, basename])), label_map[label_name]))
            
        return s3_files


class ImageNetDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.files = [
            (os.path.join(dp, f), dp.split("/")[-1])
            for dp, dn, filenames in os.walk(root_dir)
            for f in filenames
            if f.endswith(".JPEG")
        ]
        self.transform = transforms.Lambda(
            lambda img: img
        )  # No transform, just return raw image

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, label = self.files[idx]
        with open(img_path, "rb") as f:
            img = f.read()  # Read raw image bytes
        label = self.classes.index(label)
        return img, label


def scan(*, root_dir=None, epochs=10, bs=128, log_fn):
    if "s3://" in root_dir:
        # Create pytorch file loader from s3
        print("Reading from S3")
        #steps = int(np.ceil(len(list(s3_shard_urls)) / bs))        
        
        s3_dataset = S3Dataset(root_dir)
        
        data_loader = DataLoader(
            s3_dataset,
            batch_size=bs,
            num_workers=32,
            prefetch_factor=8,
            shuffle=True,
        )
    else:
        # Create pytorch file loader from filesystem
        print("Reading from a local filesystem")
        dataset = ImageNetDataset(root_dir)
        data_loader = DataLoader(
            dataset, batch_size=bs, shuffle=True, num_workers=2, prefetch_factor=8)
    
    steps = len(data_loader)

    ### Start scanning
    print(f"Batches per epoch: {steps}")

    timestamps_np = np.zeros((epochs, steps))
    batch_bytes_np = np.zeros((epochs, steps))

    first_epoch = True

    fd = open(log_fn+".csv", "w")
    fd.write("epoch,batch,batch_bytes,batch_time,timestamp,bs\n")

    for epoch in range(epochs):
        # read data for current epoch
        with tqdm(data_loader) as t:
            start_ts = time.time()
            for step, b in enumerate(t):
                images = b[0]
                labels = b[1]
                
                images_batch_bytes = sum([len(dd) for dd in images])
                labels_batch_bytes = labels.shape[0] * 4 # label is int32
                batch_bytes = images_batch_bytes + labels_batch_bytes

                batch_bytes_np[epoch][step] = batch_bytes
                timestamps_np[epoch, step] = time.time() - start_ts
                start_ts = time.time()
                
                fd.write(f"{epoch},{step},{batch_bytes},{timestamps_np[epoch, step]},{start_ts},{bs}\n")

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

    data = (bs, timestamps_np, batch_bytes_np)
    pickle.dump(data, open(log_fn, "wb"))
    fd.close()

if __name__ == "__main__":
    run(scan)
