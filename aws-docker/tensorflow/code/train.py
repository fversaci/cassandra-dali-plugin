from clize import run
from mynet import service
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tqdm import tqdm
import tensorflow as tf


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image


def train(*, bs=128, shuffle_batches=16, tfr=False):
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

    # Train the model
    model.fit(dataset, epochs=10)


if __name__ == "__main__":
    run(train)
