# Cassandra DALI Operators

The cassandra-dali-plugin provides three DALI operators for loading data from Apache Cassandra databases. Each operator is designed for specific use cases in machine learning training and inference pipelines.

## Operator Overview

| Operator | DALI Name | Primary Use Case |
|----------|-----------|------------------|
| CassandraInteractive | `fn.crs4.cassandra_interactive` | Triton inference, custom pipelines |
| CassandraSelfFeed | `fn.crs4.cassandra` | Standard training (most common) |
| CassandraDecoupled | `fn.crs4.cassandra_decoupled` | Triton inference with mini-batching |

## CassandraInteractive

The base operator that provides synchronous batch prefetching. It handles the core Cassandra connection, data retrieval, and multi-threaded loading but requires external control for feeding batches.

### When to Use

- Custom pipeline implementations
- Triton inference server integration
- When you need fine-grained control over batch feeding

### Python Wrapper

```python
from cassandra_reader_interactive import get_cassandra_reader

reader = get_cassandra_reader(
    data_table="imagenet.data_train",
    id_col="id",
    label_type="int",
    label_col="label",
    data_col="data",
    io_threads=2,
    prefetch_buffers=2,
    name="UUID",
    comm_threads=2,
    copy_threads=2,
    wait_threads=2,
    ooo=False,
    slow_start=0,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_table` | str | - | Cassandra table containing the data |
| `id_col` | str | "id" | Column name for UUID primary key |
| `label_type` | str | "int" | Label format: "int", "blob", or "none" |
| `label_col` | str | "label" | Column name for labels |
| `data_col` | str | "data" | Column name for binary data |
| `io_threads` | int | 2 | Cassandra driver IO threads (TCP connections) |
| `prefetch_buffers` | int | 2 | Multi-buffering depth for latency hiding |
| `comm_threads` | int | 2 | Communication handling threads |
| `copy_threads` | int | 2 | Data copying threads |
| `wait_threads` | int | 2 | Wait handling threads |
| `ooo` | bool | False | Out-of-order delivery for lossy networks |
| `slow_start` | int | 0 | Prefetch dilution factor |

## CassandraSelfFeed

Extends `CassandraInteractive` with self-feeding capabilities. This operator manages its own UUID list, handles shuffling, sharding for distributed training, and epoch looping. **This is the recommended operator for standard training workflows.**

### When to Use

- Standard PyTorch/TensorFlow training
- Distributed multi-GPU training
- Any training scenario requiring automatic epoch management

### Python Wrapper

The standard `get_cassandra_reader()` function in `examples/common/cassandra_reader.py` uses this operator:

```python
from cassandra_reader import get_cassandra_reader, read_uuids

# Load cached UUIDs
train_uuids = read_uuids("train.rows")

reader = get_cassandra_reader(
    data_table="imagenet.data_train",
    source_uuids=train_uuids,
    shard_id=0,
    num_shards=1,
    shuffle_every_epoch=True,
    loop_forever=True,
    prefetch_buffers=2,
    io_threads=4,
    comm_threads=1,
    copy_threads=4,
    wait_threads=2,
    ooo=False,
    slow_start=4,
)
```

### Additional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_uuids` | list | - | Full list of UUIDs to retrieve |
| `shard_id` | int | 0 | Shard index for distributed training |
| `num_shards` | int | 1 | Total number of shards |
| `shuffle_every_epoch` | bool | True | Shuffle UUIDs at each epoch |
| `loop_forever` | bool | True | Loop dataset infinitely |

### Distributed Training Example

For multi-GPU training with PyTorch:

```python
import torch
import nvidia.dali.plugin_manager as plugin_manager
from cassandra_reader import get_cassandra_reader, read_uuids

# Load UUIDs
train_uuids = read_uuids("train.rows")

# Get distributed training parameters
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()

# Create reader with sharding
reader = get_cassandra_reader(
    data_table="imagenet.data_train",
    source_uuids=train_uuids,
    shard_id=rank,
    num_shards=world_size,
    shuffle_every_epoch=True,
    loop_forever=True,
)
```

## CassandraDecoupled

A specialized operator for Triton inference server that supports mini-batch decoupling. It allows the inference server to request smaller mini-batches from a larger prefetch queue, improving throughput in serving scenarios.

### When to Use

- Triton inference server deployments
- When inference requests arrive in variable batch sizes
- High-throughput serving scenarios

### Python Wrapper

```python
from cassandra_reader_decoupled import get_cassandra_reader

reader = get_cassandra_reader(
    data_table="imagenet.data_train",
    mini_batch_size=32,
    id_col="id",
    label_type="int",
    label_col="label",
    data_col="data",
    io_threads=2,
    prefetch_buffers=2,
    name="UUID",
    comm_threads=2,
    copy_threads=2,
    wait_threads=2,
    ooo=False,
    slow_start=0,
)
```

### Additional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mini_batch_size` | int | -1 | Size of mini-batches (-1 uses full batch) |

## Triton Inference Server Integration

### Setup

The Triton examples use a two-step workflow:

1. **Prepare UUID list**: Create a JSON file with UUIDs for inference
2. **Configure model**: Use the decoupled or interactive reader in your model

### Creating UUID JSON Files

```python
from cassandra_reader_interactive import read_uuids
from crs4.cassandra_utils import get_shard
import json

def save_to_json(in_name="UUID"):
    uuids = read_uuids(rows_fn="train.rows")
    uuids, real_sz = get_shard(
        uuids,
        batch_size=128,
        shard_id=0,
        num_shards=1,
    )
    l = list()
    j = dict()
    j["data"] = l
    for u in uuids:
        b = list()
        for p in u:
            p = p.tolist()
            d = dict()
            d[in_name] = p
            b.append(d)
        l.append(b)
    with open("uuids.json", "w") as f:
        json.dump(j, f, indent=2)
```

### Triton Client Example

```python
import tritonclient.grpc as grpcclient

def start_inferring():
    client = grpcclient.InferenceServerClient(url="localhost:8001")
    
    # Prepare input data with UUIDs
    inputs = [
        grpcclient.InferInput("UUID", [batch_size, uuid_len], "UINT8")
    ]
    inputs[0].set_data_from_numpy(uuid_array)
    
    # Request inference
    results = client.infer(
        model_name="cassandra_model",
        inputs=inputs,
    )
    
    # Get outputs
    images = results.as_numpy("IMAGES")
    labels = results.as_numpy("LABELS")
```

## Performance Tuning

### Standard Training (LAN)

For typical datacenter deployments:

```python
reader = get_cassandra_reader(
    data_table="imagenet.data_train",
    prefetch_buffers=2,
    io_threads=2,
    comm_threads=2,
    copy_threads=2,
)
```

### High-Throughput Training

For maximizing throughput on fast networks:

```python
reader = get_cassandra_reader(
    data_table="imagenet.data_train",
    prefetch_buffers=4,
    io_threads=4,
    comm_threads=1,
    copy_threads=4,
)
```

### Long Fat Networks (High Latency)

For cross-datacenter or high-latency deployments (see [LFN documentation](LFN.md)):

```python
reader = get_cassandra_reader(
    data_table="imagenet.data_train",
    prefetch_buffers=16,
    io_threads=8,
    comm_threads=1,
    copy_threads=4,
    ooo=True,           # Out-of-order delivery
    slow_start=4,       # Diluted prefetching
)
```

### Parameter Guidelines

| Parameter | Low Latency | High Latency | High Bandwidth |
|-----------|-------------|--------------|----------------|
| `prefetch_buffers` | 2 | 16 | 4-8 |
| `io_threads` | 2 | 8 | 4-8 |
| `comm_threads` | 2 | 1 | 1-2 |
| `copy_threads` | 2 | 4 | 4 |
| `ooo` | False | True | False |
| `slow_start` | 0 | 4 | 0 |

## Label Types

The operators support three label formats:

### Integer Labels (Classification)

```python
reader = get_cassandra_reader(
    data_table="imagenet.data_train",
    label_type="int",
    label_col="label",
)
```

Cassandra schema:
```sql
CREATE TABLE imagenet.data_train (
    id uuid PRIMARY KEY,
    label int,
    data blob
);
```

### Image Labels (Segmentation)

```python
reader = get_cassandra_reader(
    data_table="ade20k.data_train",
    label_type="blob",
    label_col="segmentation_mask",
)
```

Cassandra schema:
```sql
CREATE TABLE ade20k.data_train (
    id uuid PRIMARY KEY,
    label blob,
    data blob
);
```

### No Labels (Inference)

```python
reader = get_cassandra_reader(
    data_table="imagenet.data_test",
    label_type="none",
)
```

## Complete Training Pipeline Example

```python
import nvidia.dali.pipeline as pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from cassandra_reader import get_cassandra_reader, read_uuids

@pipeline_def
def create_dali_pipeline(
    data_table,
    rows_fn,
    crop,
    size,
    is_training=True,
):
    # Load UUIDs
    source_uuids = read_uuids(rows_fn)
    
    # Cassandra reader
    images, labels = get_cassandra_reader(
        data_table=data_table,
        source_uuids=source_uuids,
        shuffle_every_epoch=is_training,
        prefetch_buffers=4,
        io_threads=4,
        comm_threads=1,
        copy_threads=4,
    )
    
    # Image decoding and augmentation
    dali_device = "cpu"
    decoder_device = "mixed"
    
    if is_training:
        images = fn.decoders.image_random_crop(
            images,
            device=decoder_device,
            output_type=types.RGB,
            random_aspect_ratio=[0.8, 1.25],
            random_area=[0.1, 1.0],
        )
        images = fn.resize(images, device=dali_device, resize_x=crop, resize_y=crop)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images, device=decoder_device, output_type=types.RGB)
        images = fn.resize(images, device=dali_device, size=size, mode="not_smaller")
        mirror = False
    
    images = fn.crop_mirror_normalize(
        images.gpu(),
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(crop, crop),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=mirror,
    )
    labels = labels.gpu()
    
    return images, labels
```
