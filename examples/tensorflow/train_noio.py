from clize import run
from mynet import service
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import mixed_precision
import time
import numpy as np

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
        self.fd.flush()


class FakeDataLoader(tf.keras.utils.Sequence):
    def __init__(self, num_samples, batch_size, input_shape, dtype=tf.float32):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.dtype = dtype
        self.fake_tensor = tf.constant(np.ones(input_shape), dtype=dtype)
        self.indices = np.arange(num_samples)
    
    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_size_actual = len(batch_indices)
        fake_batch = tf.stack([self.fake_tensor] * batch_size_actual)
        fake_label = tf.ones((batch_size_actual))
        return fake_batch, fake_label  # Data and label are identical

def train(*, bs=1024, epochs=4, num_samples=153600, log_fn=None, multigpu=False):
    # tf dataservice expects the global batchsize
    if multigpu:
        num_gpus = len(tf.config.experimental.list_physical_devices("GPU"))
    else:
        num_gpus = 1

    bs = bs * num_gpus
    
    print (f"GPUs: {num_gpus}")
    print (f"Global batch size: {bs}")

    # Fake dataloader
    dataset = FakeDataLoader(num_samples, bs, input_shape=(224,224,3))
    
    # Set precision
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)
    
    # Multi-GPU strategy
    if multigpu:
        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            # Define the model
            model = ResNet50(
                weights=None, input_shape=(224, 224, 3), classes=1000
            ) 
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
    else:
        # Define the model
        model = ResNet50(
            weights=None, input_shape=(224, 224, 3), classes=1000
        ) 
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
