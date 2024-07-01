# pip install tqdm

import io
import os
import random
import tensorflow as tf
from tqdm import tqdm
import numpy as np


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image_to_tfexample(image_data, label):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/encoded": _bytes_feature(image_data),
                "image/class/label": _int64_feature(label),
            }
        )
    )


def load_image(image_path):
    """
    Loads an image from the given path and returns it as a byte array if it is a proper JPEG.
    """
    try:
        with open(image_path, "rb") as fh:
            image_data = fh.read()

        # Decode the image to check if it is a valid JPEG
        image = tf.io.decode_jpeg(image_data)

        # If decoding succeeds, return the image data
        return image_data
    except Exception as e:
        # If an error occurs during decoding, return None
        print(f"Error decoding image {image_path}: {e}")
        return None


def create_label_map(data_dir):
    label_map = {}
    for idx, label in enumerate(sorted(os.listdir(data_dir))):
        label_map[label] = idx
    return label_map


def write_tfrecords(data_dir, output_dir, split_name, max_file_size=64 * 1024 * 1024):
    output_dir = os.path.join(output_dir, split_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    label_map = create_label_map(data_dir)

    file_index = 0
    current_file_size = 0
    tfrecord_path = os.path.join(output_dir, f"{split_name}_{file_index}.tfrecord")
    writer = tf.io.TFRecordWriter(tfrecord_path)

    print("Scanning images")
    all_image_paths = []
    for label in tqdm(label_map.keys()):
        class_dir = os.path.join(data_dir, label)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            all_image_paths.append((image_path, label))
    print("Shuffling")
    random.shuffle(all_image_paths)

    print("Writing tfrecords")
    for image_path, label in tqdm(all_image_paths):
        try:
            image_data = load_image(image_path)
            if image_data is None:
                continue

            label_int = label_map[label]
            example = image_to_tfexample(image_data, label_int)
            serialized_example = example.SerializeToString()

            # Check if adding this example exceeds the max_file_size
            if current_file_size + len(serialized_example) > max_file_size:
                writer.close()
                file_index += 1
                current_file_size = 0
                tfrecord_path = os.path.join(
                    output_dir, f"{split_name}_{file_index}.tfrecord"
                )
                writer = tf.io.TFRecordWriter(tfrecord_path)

            writer.write(serialized_example)
            current_file_size += len(serialized_example)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    writer.close()


if __name__ == "__main__":
    train_data_dir = "/data/imagenet/train_orig"
    val_data_dir = "/data/imagenet/val_orig"
    output_dir = "/data/imagenet/tfrecords"

    write_tfrecords(val_data_dir, output_dir, "val")
    write_tfrecords(train_data_dir, output_dir, "train")
