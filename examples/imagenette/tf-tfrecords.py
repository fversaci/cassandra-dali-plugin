import tensorflow as tf
import os
from tqdm import tqdm

# import boto3


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


def get_tfrecord_lengths(tfrecord_paths, bs, prefetch=8):
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

    for b_i, b_l in tqdm(dataset):
        ...

    # for encoded_image, label in dataset:
    #     size = tf.strings.length(encoded_image)
    #     print(f"{size}, {label}")


def scan_directory(directory_path):
    tfrecord_paths = [
        os.path.join(directory_path, filename)
        for filename in os.listdir(directory_path)
        if filename.endswith(".tfrecord")
    ]
    print(f"Found {len(tfrecord_paths)} TFRecord files.")
    return tfrecord_paths


def main():
    # Disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Set the directory path
    directory_path = "/data/imagenet/tfrecords/val"
    paths = scan_directory(directory_path)
    get_tfrecord_lengths(paths, bs=128)

    # directory_path = 's3://imagenette/tfrecords/train'
    # paths = list_s3_files(directory_path)
    # get_tfrecord_lengths(paths)


if __name__ == "__main__":
    main()
