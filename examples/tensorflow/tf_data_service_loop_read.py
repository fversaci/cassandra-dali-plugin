import tensorflow as tf
from tqdm import tqdm
from mynet import service
from clize import run


def main(*, bs=1024, shuffle_batches=16, tfr=False):
    # Choose if files or tfrecords
    if tfr:
        dataset_id="imagenet_tfr"
    else:
        dataset_id="imagenet_files"
    # Distribute the dataset using tf.data service
    dataset = tf.data.experimental.service.from_dataset_id(
        processing_mode="parallel_epochs",
        service=service,
        dataset_id=dataset_id,
        element_spec=(tf.TensorSpec(shape=(None,), dtype=tf.string, name=None), tf.TensorSpec(shape=(None,), dtype=tf.int64, name=None))
    )

    # Shuffle, batch, and prefetch
    dataset = dataset.unbatch()
    dataset = dataset.shuffle(buffer_size=bs*shuffle_batches)
    dataset = dataset.batch(bs)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    for _ in tqdm(dataset):
        ...


if __name__ == "__main__":
    run(main)
