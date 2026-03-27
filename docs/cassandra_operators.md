# Cassandra DALI Operators

This document describes the three DALI operators provided by the cassandra-dali-plugin for loading image and binary data from Apache Cassandra databases into ML training and inference pipelines.

## Overview

The plugin provides three operators, each optimized for different use cases:

| Operator                 | DALI Function Name              | Primary Use Case                          |
|--------------------------|---------------------------------|-------------------------------------------|
| **CassandraSelfFeed**    | `fn.crs4.cassandra`             | Standard training workflows (recommended) |
| **CassandraInteractive** | `fn.crs4.cassandra_interactive` | Custom pipelines, Triton inference        |
| **CassandraDecoupled**   | `fn.crs4.cassandra_decoupled`   | Triton inference with mini-batching       |

### Choosing an Operator

- **For standard training**: Use `CassandraSelfFeed` (`fn.crs4.cassandra`). It handles UUID management, shuffling, sharding, and epoch looping automatically.
- **For Triton inference**: Use `CassandraInteractive` for basic inference or `CassandraDecoupled` for mini-batch decoupling.
- **For custom pipelines**: Use `CassandraInteractive` when you need fine-grained control over batch feeding.

## Prerequisites

Before using these operators, ensure you have:

1. **Installed the plugin**:
   ```bash
   pip3 install . --no-build-isolation
   ```

2. **Loaded the plugin library**:
   ```python
   import crs4.cassandra_utils
   import nvidia.dali.plugin_manager as plugin_manager
   import pathlib

   plugin_path = pathlib.Path(crs4.cassandra_utils.__path__[0])
   plugin_path = plugin_path.parent.parent.joinpath("libcrs4cassandra.so")
   plugin_manager.load_library(str(plugin_path))
   ```

3. **Configured Cassandra credentials** in `private_data.py` (see `examples/common/private_data.template.py`)

## CassandraSelfFeed

The recommended operator for standard training workflows. It extends `CassandraInteractive` with self-feeding capabilities, managing UUID lists, shuffling, sharding for distributed training, and epoch looping automatically.

### When to Use

- Standard PyTorch/TensorFlow training
- Distributed multi-GPU training
- Any training scenario requiring automatic epoch management

### Python Wrapper

The standard `get_cassandra_reader()` function in `examples/common/cassandra_reader.py` uses this operator:

```python
from cassandra_reader import get_cassandra_reader, read_uuids

# Load cached UUIDs from metadata table
train_uuids = read_uuids("train.rows")

# Create reader
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

### Parameters

#### Connection Parameters

| Parameter    | Type | Default    | Description                                                         |
|--------------|------|------------|---------------------------------------------------------------------|
| `data_table` | str  | *required* | Cassandra table containing the data (e.g., `"imagenet.data_train"`) |
| `id_col`     | str  | `"id"`     | Column name for UUID primary key                                    |
| `data_col`   | str  | `"data"`   | Column name for binary data (BLOB)                                  |

#### Label Parameters

| Parameter    | Type | Default   | Description                                                                              |
|--------------|------|-----------|------------------------------------------------------------------------------------------|
| `label_type` | str  | `"int"`   | Label format: `"int"` (classification), `"blob"` (segmentation), or `"none"` (inference) |
| `label_col`  | str  | `"label"` | Column name for labels                                                                   |

#### Dataset Management Parameters

| Parameter             | Type | Default    | Description                                          |
|-----------------------|------|------------|------------------------------------------------------|
| `source_uuids`        | list | *required* | Full list of UUIDs to retrieve (from `read_uuids()`) |
| `shard_id`            | int  | `0`        | Shard index for this process in distributed training |
| `num_shards`          | int  | `1`        | Total number of shards (processes)                   |
| `shuffle_every_epoch` | bool | `True`     | Shuffle UUIDs at the start of each epoch             |
| `loop_forever`        | bool | `True`     | Loop dataset infinitely (set `False` for validation) |

#### Performance Parameters

| Parameter          | Type | Default | Description                                                     |
|--------------------|------|---------|-----------------------------------------------------------------|
| `prefetch_buffers` | int  | `2`     | Multi-buffering depth for latency hiding                        |
| `io_threads`       | int  | `2`     | Cassandra driver IO threads (limits TCP connections)            |
| `comm_threads`     | int  | `2`     | Communication handling threads                                  |
| `copy_threads`     | int  | `2`     | Data copying threads                                            |
| `wait_threads`     | int  | `2`     | Wait handling threads                                           |
| `ooo`              | bool | `False` | Out-of-order delivery for high-latency/lossy networks           |
| `slow_start`       | int  | `0`     | Prefetch dilution factor (request extra image every N requests) |

### Distributed Training Example

For multi-GPU training with PyTorch:

```python
import torch
import torch.distributed as dist
from cassandra_reader import get_cassandra_reader, read_uuids

# Initialize distributed training
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Load UUIDs (same file on all ranks)
train_uuids = read_uuids("train.rows")

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

## CassandraInteractive

The base operator that provides synchronous batch prefetching. It handles Cassandra connection, data retrieval, and multi-threaded loading but requires external control for feeding batches.

### When to Use

- Custom pipeline implementations requiring manual batch control
- Triton inference server integration
- When you need fine-grained control over when batches are fetched

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

This operator uses the same connection, label, and performance parameters as `CassandraSelfFeed`, but does **not** support the dataset management parameters (`source_uuids`, `shard_id`, `num_shards`, `shuffle_every_epoch`, `loop_forever`).

| Parameter | Type | Default  | Description                     |
|-----------|------|----------|---------------------------------|
| `name`    | str  | `"UUID"` | Operator name for DALI pipeline |

## CassandraDecoupled

A specialized operator for Triton inference server that supports mini-batch decoupling. It allows the inference server to request smaller mini-batches from a larger prefetch queue, improving throughput in serving scenarios.

### When to Use

- Triton inference server deployments
- When inference requests arrive with variable batch sizes
- High-throughput serving scenarios requiring flexible batching

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

| Parameter         | Type | Default | Description                                      |
|-------------------|------|---------|--------------------------------------------------|
| `mini_batch_size` | int  | `-1`    | Size of mini-batches (`-1` uses full batch size) |

## Label Types

The operators support three label formats for different ML tasks:

### Integer Labels (Classification)

For image classification tasks where labels are class indices:

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

For semantic segmentation tasks where labels are pixel-wise masks:

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

For inference workloads where only input data is needed:

```python
reader = get_cassandra_reader(
    data_table="imagenet.data_test",
    label_type="none",
)
```

## Performance Tuning

### Standard Training (Low-Latency Network)

For typical datacenter deployments with low-latency connections:

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

| Scenario        | `prefetch_buffers` | `io_threads` | `comm_threads` | `copy_threads` | `ooo` | `slow_start` |
|-----------------|--------------------|--------------|----------------|----------------|-------|--------------|
| Low latency     | 2                  | 2            | 2              | 2              | False | 0            |
| High throughput | 4                  | 4            | 1              | 4              | False | 0            |
| High latency    | 16                 | 8            | 1              | 4              | True  | 4            |
| Packet loss     | 16                 | 8            | 1              | 4              | True  | 4            |

### Out-of-Order Delivery (`ooo`)

When `ooo=True`, images are returned as soon as they arrive from Cassandra, potentially altering their sequence and mixing different batches. This is beneficial for:

- High-latency networks where packet delays can stall the pipeline
- Lossy networks where retransmissions cause variable delays
- Scenarios where batch ordering is less important than throughput

**Note**: Enable this only when your training loop can handle non-sequential batches.

### Prefetch Dilution (`slow_start`)

The `slow_start` parameter controls diluted prefetching to limit initial request bursts:

- `slow_start=0`: Normal prefetching (default)
- `slow_start=4`: Request an extra image every 4 normal requests

This helps prevent packet loss on networks where initial bursts can overwhelm routers.

## Triton Inference Server Integration

### Creating UUID JSON Files

For Triton inference, prepare a JSON file with UUIDs:

```python
from cassandra_reader_interactive import read_uuids
from crs4.cassandra_utils import get_shard
import json

def save_to_json(output_file="uuids.json", batch_size=128):
    uuids = read_uuids(rows_fn="train.rows")
    uuids, real_sz = get_shard(
        uuids,
        batch_size=batch_size,
        shard_id=0,
        num_shards=1,
    )
    
    data = {"data": []}
    for batch in uuids:
        batch_list = []
        for uuid_tensor in batch:
            batch_list.append({"UUID": uuid_tensor.tolist()})
        data["data"].append(batch_list)
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

save_to_json()
```

### Triton Client Example

```python
import numpy as np
import tritonclient.grpc as grpcclient

def run_inference(url="localhost:8001", model_name="cassandra_model"):
    client = grpcclient.InferenceServerClient(url=url)
    
    # Prepare UUID input
    uuid_array = np.array(uuid_list, dtype=np.uint8)
    inputs = [
        grpcclient.InferInput("UUID", uuid_array.shape, "UINT8")
    ]
    inputs[0].set_data_from_numpy(uuid_array)
    
    # Request inference
    results = client.infer(
        model_name=model_name,
        inputs=inputs,
    )
    
    # Get outputs
    images = results.as_numpy("IMAGES")
    labels = results.as_numpy("LABELS")
    
    return images, labels
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
    prefetch_buffers=4,
    io_threads=4,
):
    # Load UUIDs
    source_uuids = read_uuids(rows_fn)
    
    # Cassandra reader
    images, labels = get_cassandra_reader(
        data_table=data_table,
        source_uuids=source_uuids,
        shuffle_every_epoch=is_training,
        loop_forever=is_training,
        prefetch_buffers=prefetch_buffers,
        io_threads=io_threads,
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
            num_attempts=100,
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


# Create and build pipeline
pipe = create_dali_pipeline(
    data_table="imagenet.data_train",
    rows_fn="train.rows",
    crop=224,
    size=256,
    is_training=True,
    batch_size=64,
    num_threads=4,
    device_id=0,
)
pipe.build()
```
