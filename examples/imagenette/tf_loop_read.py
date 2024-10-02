import os, sys
import tensorflow as tf
from clize import run
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from IPython import embed
import time
import pickle

def parse_image(img, label):
    return tf.io.decode_jpeg(img), label

## tfrecords

def get_tfrecord_dataset(tfrecord_paths, bs=128, prefetch=8):
    # Function to parse each TFRecord
    def parse_tfrecord_fn(example):
        features = {
            "image/encoded": tf.io.FixedLenFeature([], tf.string, ""),
            "image/class/label": tf.io.FixedLenFeature([], tf.int64, -1),
        }
        example = tf.io.parse_single_example(example, features)
        encoded_image = example["image/encoded"]
        label = example["image/class/label"]
        return encoded_image, label

    # Create an empty dataset
    dataset = tf.data.Dataset.from_tensor_slices(tfrecord_paths)

    # Define cycle_length and block_length to control the parallel reading of files
    cycle_length = 32  # Number of files to read in parallel

    # Interleave TFRecord files to parallelize reads
    dataset = dataset.interleave(
        lambda filepath: tf.data.TFRecordDataset(filepath).map(
            parse_tfrecord_fn,  # num_parallel_calls=tf.data.experimental.AUTOTUNE
        ),
        cycle_length=cycle_length,  # Number of input elements that will be processed concurrently
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    # Add prefetching
    epochs = 30
    dataset = dataset.repeat(epochs).shuffle(1024).batch(bs).prefetch(prefetch)
    
    return dataset

def scan_directory(directory_path):
    tfrecord_paths = [
        os.path.join(directory_path, filename)
        for filename in os.listdir(directory_path)
        if filename.endswith(".tfrecord")
    ]
    print(f"Found {len(tfrecord_paths)} TFRecord files.")
    return tfrecord_paths

## tf-data files
class ImageNetDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.files = [
            (os.path.join(dp, f), dp.split("/")[-1])
            for dp, dn, filenames in os.walk(root_dir)
            for f in filenames
            if f.endswith(".JPEG")
        ]

    def _generator(self):
        for img_path, label in self.files:
            with open(img_path, "rb") as f:
                img = f.read()  # Read raw image bytes
            label = self.classes.index(label)
            yield img, label

    def get_dataset(self):
        return tf.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )  # .map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)


## Loop read
def scan(*, root_dir="", tfr=False, epochs=4, bs=128, log_fn=None):
    print("Reading from a local filesystem")
    
    if tfr:
        paths = scan_directory(root_dir)
        dataset = get_tfrecord_dataset(paths, bs=bs, prefetch=8)
    else:
        dataset = (
            ImageNetDataset(root_dir).get_dataset().batch(bs).prefetch(tf.data.AUTOTUNE)
        )
       
    ## Get the number of batches looping across the whole dataset
    steps = 0
    for _ in tqdm(dataset):
        steps += 1

    ### Start scanning
    print(f"Batches per epoch: {steps}")

    timestamps_np = np.zeros((epochs, steps))
    batch_bytes_np = np.zeros((epochs, steps))

    first_epoch = True

    for epoch in range(epochs):
        # read data for current epoch
        with tqdm(dataset) as t:
            start_ts = time.time()
            for step, b in enumerate(t):
                images = b[0]
                labels = b[1]
               
                #if step == 10:
                #    embed()
                
                images_batch_bytes = sum([len(dd.numpy()) for dd in images])
                labels_batch_bytes = labels.shape[0] * 4
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
