import os, sys
import tensorflow as tf
from clize import run
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from IPython import embed
import time
import pickle


## tf-data files


def parse_image(filename, label):
    img = tf.io.read_file(filename)
    # img = tf.io.decode_jpeg(img, channels=3)
    return img, label


def get_raw_image_dataset(root_dir, bs=128, prefetch=8, shuffle_batches=16):
    # List all JPEG files recursively
    dataset = tf.data.Dataset.list_files(os.path.join(root_dir, "*/*.JPEG"))

    # Infer labels from file paths
    def extract_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]  # Assumes that the label is the parent directory
        return label

    dataset = dataset.map(
        lambda x: (x, extract_label(x)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Convert string labels to integer indices
    class_names = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
    class_indices = dict((name, idx) for idx, name in enumerate(class_names))

    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(class_names),
            values=tf.constant(list(range(len(class_names))), dtype=tf.int64),
        ),
        default_value=-1,
    )

    dataset = dataset.map(
        lambda x, y: (x, table.lookup(y)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Parse and preprocess images
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle, batch, and prefetch
    dataset = dataset.shuffle(buffer_size=bs*shuffle_batches)
    dataset = dataset.batch(bs)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


## tfrecords


def get_tfrecord_dataset(tfrecord_paths, bs=128, prefetch=8, shuffle_batches=16):
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
            parse_tfrecord_fn,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        ),
        cycle_length=cycle_length,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    # Add prefetching
    dataset = dataset.shuffle(bs*shuffle_batches).batch(bs).prefetch(prefetch)

    return dataset


def scan_directory(directory_path):
    tfrecord_paths = [
        os.path.join(directory_path, filename)
        for filename in os.listdir(directory_path)
        if filename.endswith(".tfrecord")
    ]
    print(f"Found {len(tfrecord_paths)} TFRecord files.")
    return tfrecord_paths


## Loop read
def scan(*, root_dir="", tfr=False, epochs=4, bs=128, shuffle_batches=16, log_fn):
    print("Reading from a local filesystem")

    if tfr:
        paths = scan_directory(root_dir)
        dataset = get_tfrecord_dataset(paths, bs=bs, prefetch=8, shuffle_batches=shuffle_batches)
        ## Get the number of batches looping across the whole dataset
        steps = 0
        for _ in tqdm(dataset):
            steps += 1
    else:
        dataset = get_raw_image_dataset(root_dir, bs=bs, prefetch=8, shuffle_batches=shuffle_batches)
        steps = tf.data.experimental.cardinality(dataset).numpy()

    ### Start scanning
    print(f"Batches per epoch: {steps}")

    timestamps_np = np.zeros((epochs, steps))
    batch_bytes_np = np.zeros((epochs, steps))

    first_epoch = True
    
    if log_fn:
        fd = open(log_fn+".csv", "w")
        fd.write("epoch,batch,batch_bytes,batch_time\n")

    for epoch in range(epochs):
        # read data for current epoch
        with tqdm(dataset) as t:
            start_ts = time.time()
            for step, b in enumerate(t):
                images = b[0].numpy()
                labels = b[1]

                # if step == 10:
                #    embed()

                images_batch_bytes = sum([len(dd) for dd in images])
                labels_batch_bytes = labels.shape[0] * 4
                batch_bytes = images_batch_bytes + labels_batch_bytes

                batch_bytes_np[epoch][step] = batch_bytes
                timestamps_np[epoch, step] = time.time() - start_ts
                start_ts = time.time()

                fd.write(f"{epoch},{step},{batch_bytes},{timestamps_np[epoch, step]}\n")

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
        fd.close()

if __name__ == "__main__":
    run(scan)
