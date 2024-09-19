import os
import tensorflow as tf
from tqdm import tqdm
import numpy as np


def parse_image(img, label):
    return tf.io.decode_jpeg(img), label


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


def scan(root_dir="/data/imagenet/train", epochs=4):
    dataset = (
        ImageNetDataset(root_dir).get_dataset().batch(128).prefetch(tf.data.AUTOTUNE)
    )
    speeds = []
    first_epoch = True
    for _ in range(epochs):
        t = tqdm(dataset)
        for _ in t:
            ...
        epoch_time = t.format_dict["elapsed"]
        if first_epoch:
            first_epoch = False
        else:
            speeds.append(t.total / epoch_time)

    if epochs > 3:
        average_speed = np.mean(speeds)
        std_dev_speed = np.std(speeds)
        print(f"Stats for epochs > 1")
        print(f"  Average speed: {average_speed:.2f} ± {std_dev_speed:.2f} it/s")


if __name__ == "__main__":
    scan()
