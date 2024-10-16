from clize import run
from mynet import worker_address
import os
import tensorflow as tf


## read files


def parse_image(filename, label):
    img = tf.io.read_file(filename)
    # img = tf.io.decode_jpeg(img, channels=3)
    return img, label


def get_raw_image_dataset(root_dir):
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

    return dataset


## read tfrecords


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


def get_tfrecord_dataset(tfrecord_paths):

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

    return dataset


def scan_directory(directory_path):
    tfrecord_paths = [
        os.path.join(directory_path, filename)
        for filename in os.listdir(directory_path)
        if filename.endswith(".tfrecord")
    ]
    print(f"Found {len(tfrecord_paths)} TFRecord files.")
    return tfrecord_paths


## tf.data service


def start_server():
    dconf = tf.data.experimental.service.DispatcherConfig(port=5050)
    dispatcher = tf.data.experimental.service.DispatchServer(dconf)
    dispatcher_address = dispatcher.target.split("://")[1]
    wconf = tf.data.experimental.service.WorkerConfig(
        dispatcher_address=dispatcher_address,
        data_transfer_address="0.0.0.0",
        worker_address=worker_address,
        port=5051,
    )
    worker = tf.data.experimental.service.WorkerServer(wconf)
    print("Target: ", dispatcher.target)
    return worker, dispatcher


## main


def main(*, bs=128):
    # setup datasets dataset
    root_dir = "/data/imagenet-tfrecords/train/"
    paths = scan_directory(root_dir)
    dataset_tfr = get_tfrecord_dataset(paths)
    dataset_files = get_raw_image_dataset("/data/imagenet-files/train/")

    # Shuffle, batch, and prefetch
    dataset_files = dataset_files.batch(bs)
    dataset_files = dataset_files.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset_tfr = dataset_tfr.batch(bs)
    dataset_tfr = dataset_tfr.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    # Distribute the dataset using tf.data service
    worker, dispatcher = start_server()
    dataset_id = tf.data.experimental.service.register_dataset(
        dispatcher.target, dataset_files, dataset_id="imagenet_files"
    )
    dataset_id = dataset_id.numpy().decode("utf-8")
    print("Registered dataset: ", dataset_id)
    print("Element specification: ", dataset_files.element_spec)
    dataset_id = tf.data.experimental.service.register_dataset(
        dispatcher.target, dataset_tfr, dataset_id="imagenet_tfr"
    )
    dataset_id = dataset_id.numpy().decode("utf-8")
    print("Registered dataset: ", dataset_id)
    print("Element specification: ", dataset_tfr.element_spec)
    worker.join()


if __name__ == "__main__":
    run(main)
