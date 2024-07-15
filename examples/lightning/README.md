# Imagenette Dataset with Lightning
## Starting and filling the DB

Once the Docker container is set up, it is possible to start the
database and populate it with images from the imagenette dataset using
the provided script within the Docker container:

```bash
./start-and-fill-db.sh  # might take a few minutes
```

In this example we will import the [Imagenette2-320
dataset](https://github.com/fastai/imagenette) (a subset of ImageNet)
as a Cassandra dataset and then read the data into NVIDIA DALI.

The raw files are already present in the `/tmp` directory of the
provided [Docker container](../../README.md#running-the-docker-container),
from which the following commands can be run.

## Multi-GPU training

Run the training of the Imagenette dataset with [the lightning application](train_model.py) with:
```bash
$ python3 train_model.py --num-gpu NUM_GPUS \
  -a resnet50 --b 64 --workers 4 --lr=1.0e-3 \
  --train-data-table imagenette.data_train_orig --train-metadata-table imagenette.metadata_train_orig \
  --val-data-table imagenette.data_val_orig --val-metadata-table imagenette.metadata_val_orig
```
