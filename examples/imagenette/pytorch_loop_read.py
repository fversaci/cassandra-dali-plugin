import os
import statistics
from clize import run
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm, trange

from torchdata.datapipes.iter import IterableWrapper
from IPython import embed

import time
import numpy as np
import pickle


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


def custom_collate(batch):
    return batch


def scan(*, root_dir="s3://imagenet/val", epochs=4, bs=128, log_fn=None):
    if "s3://" in root_dir:
        # Create pytorch file loader from s3
        print("Reading from S3")
        s3_shard_urls = IterableWrapper(
            [
                root_dir,
            ]
        ).list_files_by_s3()
        s3_files = s3_shard_urls.sharding_filter().load_files_by_s3()
        # tf_records = s3_files.load_from_tfrecord()
        # text data
        data_loader = DataLoader(
            s3_files,
            batch_size=bs,
            num_workers=32,
            prefetch_factor=8,
            collate_fn=custom_collate,
            shuffle=True,
        )

    else:
        # Create pytorch file loader from filesystem
        print("Reading from a local filesystem")
        dataset = ImageNetDataset(root_dir)
        data_loader = DataLoader(
            dataset, batch_size=bs, shuffle=True, num_workers=2, prefetch_factor=8
        )

    ### Start scanning
    steps = len(data_loader)
    print(f"Batches per epoch: {steps}")

    timestamps_np = np.zeros((epochs, steps))
    batch_bytes_np = np.zeros((epochs, steps))

    first_epoch = True

    for epoch in range(epochs):
        # read data for current epoch
        with tqdm(data_loader) as t:
            start_ts = time.time()
            for step, b in enumerate(t):
                images = b[0]
                labels = b[1]

                images_batch_bytes = sum([len(dd) for dd in images])
                labels_batch_bytes = labels.shape[0]
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


if __name__ == "__main__":
    run(scan)
