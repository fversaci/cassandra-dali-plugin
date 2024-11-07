from clize import run
from mynet import service
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import mixed_precision
import time

from tensorflow.keras.callbacks import Callback


class BatchEpochTimeLogger(Callback):
    def __init__(self, log_fd):
        super().__init__()
        self.fd = log_fd

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self, "epoch"):
            self.fd.write(
                "epoch,step,train_acc1,train_acc5,train_batch_ts,train_loss,val_acc1,val_acc5,val_batch_ts,val_loss\n"
            )

        self.epoch = epoch

    def on_train_batch_begin(self, batch, logs=None):
        ts = time.time()
        self.fd.write(f"{self.epoch},{batch},,,{ts},,,,,\n")


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.resnet50.preprocess_input(image)
    # image = tf.keras.applications.imagenet_utils.preprocess_input(image, mode="torch")
    return image


def train(*, bs=1024, epochs=4, shuffle_batches=16, tfr=False, log_fn=None):

    # tf dataservice expects the global batchsize
    num_gpus = len(tf.config.experimental.list_physical_devices("GPU"))
    bs = bs * num_gpus

    # Choose if files or tfrecords
    if tfr:
        dataset_id = "imagenet_tfr"
    else:
        dataset_id = "imagenet_files"
    # Distribute the dataset using tf.data service
    dataset = tf.data.experimental.service.from_dataset_id(
        processing_mode="parallel_epochs",
        service=service,
        dataset_id=dataset_id,
        element_spec=(
            tf.TensorSpec(shape=(None,), dtype=tf.string, name=None),
            tf.TensorSpec(shape=(None,), dtype=tf.int64, name=None),
        ),
    )

    # Shuffle, batch, and prefetch
    dataset = dataset.unbatch()
    dataset = dataset.map(
        lambda x, y: (preprocess_image(x), y),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.shuffle(buffer_size=bs * shuffle_batches)
    dataset = dataset.batch(bs)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Multi-GPU strategy
    strategy = tf.distribute.MirroredStrategy()

    # Set precision
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)

    with strategy.scope():
        # Define the model
        model = ResNet50(
            weights=None, input_shape=(224, 224, 3), classes=1000
        )  # Adjust number of classes
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    if log_fn:
        fd = open(log_fn, "w")
        time_logger = BatchEpochTimeLogger(fd)
        print("Start training with log")
        model.fit(dataset, epochs=epochs, callbacks=[time_logger])
        fd.close()
    else:
        # Train the model
        print("Start training")
        model.fit(dataset, epochs=epochs)


if __name__ == "__main__":
    run(train)
