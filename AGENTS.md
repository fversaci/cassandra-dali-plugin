# Cassandra-DALI Plugin - Agent Guide

## Project Overview

NVIDIA DALI plugin for loading image/binary data from Apache Cassandra database into ML training pipelines. Tested with DALI v1.53.

**Repository**: https://github.com/crs4/cassandra-dali-plugin
**Version**: 1.3.0 (from pyproject.toml)
**License**: Apache License 2.0
**Python**: >=3.11,<3.14
**Authors**: Francesco Versaci, Giovanni Busonera (CRS4)

## Code Organization

```
crs4/
├── cpp/                          # C++ DALI plugin source
│   ├── CMakeLists.txt            # CMake build (invoked via setup.py)
│   ├── ThreadPool.h              # ThreadPool (3rd party, see below)
│   ├── batch_loader.cc/h         # Batch loading logic
│   ├── cassandra_dali_interactive.cc/h  # Interactive reader
│   ├── cassandra_dali_selffeed.cc/h     # Self-feed reader
│   └── cassandra_dali_decoupled.cc/h    # Decoupled reader (Triton)
└── cassandra_utils/              # Python utilities
    ├── __init__.py               # Package exports (re-exports all classes)
    ├── _cassandra_config.py      # CassandraConf dataclass
    ├── _cassandra_session.py     # Cassandra session management
    ├── _cassandra_writer.py      # Base writer class
    ├── _cassandra_classification_writer.py
    ├── _cassandra_segmentation_writer.py
    ├── _list_manager.py          # UUID list management
    ├── _mini_list_manager.py
    ├── _sharding.py              # Sharding utilities (get_shard)
    └── _split_generator.py       # Split file generation

examples/
├── common/                        # Shared utilities (all examples depend on this)
│   ├── cassandra_reader.py       # DALI reader wrapper + plugin loader
│   ├── private_data.template.py  # Template for credentials (copy to private_data.py)
│   ├── fn_shortcuts.py           # DALI function shortcuts
│   ├── extract_common.py         # Common extraction utilities (used by Spark jobs)
│   ├── cache_uuids.py            # Cache UUIDs from metadata table to .rows file
│   └── extract_serial.py         # Serial data loader (no Spark)
├── imagenette/                   # Classification example
│   ├── create_tables.cql         # Cassandra schema
│   ├── extract_spark.py          # Spark-based data loader
│   ├── extract_serial.py         # Serial data loader
│   ├── cache_uuids.py            # UUID caching helper
│   ├── loop_read.py             # DALI reading test script
│   ├── distrib_train_from_cassandra.py  # Multi-GPU training
│   ├── distrib_train_from_file.py      # Original DALI file-based training
│   └── create_tfrecord.py        # TFRecord conversion
├── lightning/                    # PyTorch Lightning variant of imagenette
├── ade20k/                       # Segmentation example
├── corel5k/                      # Multilabel example
├── splitfile/                    # Split-file generation
└── triton/                       # Triton inference server

docker-scripts/                   # Test scripts (run inside container)
  ├── test-imagenette.sh
  ├── test-lightning.sh
  ├── test-ade20k.sh
  ├── test-corel5k.sh
  └── test-splitfile.sh

docker-compose.yml               # Cassandra + DALI containers
docker-compose.triton.yml        # Triton inference variant
Dockerfile.dali-cassandra         # DALI client container (NGC 26.02-py3)
Dockerfile.cassandra              # Cassandra server container
Dockerfile.dali-cassandra-triton  # DALI + Triton container
setup.py                          # Python package with CMakeExtension
```

## Essential Commands

### Build & Install

```bash
# Build and install plugin (compiles C++ via CMake under the hood)
# The Cassandra C++ driver is built automatically if not found.
pip install .
```

### Docker Workflow

```bash
# Start services
docker compose up --build -d

# Wait 1-2 minutes for Cassandra to be ready, then enter container
docker compose exec dali-cassandra fish

# Run example tests (inside container)
./docker-scripts/test-imagenette.sh
./docker-scripts/test-lightning.sh
./docker-scripts/test-ade20k.sh
```

The test scripts drop/recreate the Cassandra keyspace, load data via Spark (or serial), then run read tests including GPU reads and a full training epoch.

### Development Workflow Inside Container

```bash
# Rebuild plugin after code changes
pip install . --no-build-isolation

# Run a single example manually
cd examples/imagenette
python3 cache_uuids.py --metadata-table=imagenette.metadata_train --rows-fn train.rows
python3 loop_read.py --data-table imagenette.data_train --rows-fn train.rows
python3 loop_read.py --data-table imagenette.data_train --rows-fn train.rows --use-gpu

# Multi-GPU training test (1 epoch)
torchrun --nproc_per_node=NUM_GPUS distrib_train_from_cassandra.py \
  -a resnet50 --dali_cpu --b 64 --loss-scale 128.0 --workers 4 --lr=0.4 --opt-level O2 --epochs 1 \
  --train-data-table imagenette.data_train --train-rows-fn train.rows \
  --val-data-table imagenette.data_val --val-rows-fn val.rows
```

### Triton Inference

```bash
docker compose -f docker-compose.triton.yml up --build -d
```

### Spark-Based Data Loading (for large datasets)

```bash
# Start Spark master+worker (inside container)
/spark/sbin/start-master.sh
/spark/sbin/start-worker.sh spark://$HOSTNAME:7077

# Load data in parallel with Spark
/spark/bin/spark-submit --master spark://$HOSTNAME:7077 --conf spark.default.parallelism=10 \
  --py-files extract_common.py extract_spark.py /data/imagenet/ \
  --split-subdir=train --data-table imagenet.data_train --metadata-table imagenet.metadata_train
```

### UUID Caching Workflow

The examples use a two-phase workflow:

1. **Cache UUIDs** from the metadata table to a `.rows` file (pickle format):
   ```bash
   python3 cache_uuids.py --metadata-table=imagenette.metadata_train --rows-fn train.rows
   ```

2. **Read data** using the cached UUID list:
   ```bash
   python3 loop_read.py --data-table imagenette.data_train --rows-fn train.rows
   ```

The `.rows` files contain a pickled dict with `row_keys` (list of UUID strings). This separation allows filtering the metadata (e.g., by label split) once, then reusing the cached UUID list for multiple training runs.

## Build System

### C++ Plugin (CMake)

| Property           | Value                                  |
|--------------------|----------------------------------------|
| Minimum CMake      | 3.25.2                                 |
| C++ Standard       | C++20                                  |
| CUDA Standard      | C++20 (`-std=c++20` flag)              |
| CUDA Architectures | 75;80;86;89;90                         |
| Output             | `libcrs4cassandra.so` (shared library) |

CMake queries DALI at configure time for include paths and library directories:

```cmake
execute_process(
    COMMAND python3 -c "import nvidia.dali as dali; print(dali.sysconfig.get_lib_dir())"
    OUTPUT_VARIABLE DALI_LIB_DIR)
execute_process(
    COMMAND python3 -c "import nvidia.dali as dali; print(\" \".join(dali.sysconfig.get_compile_flags()))"
    OUTPUT_VARIABLE DALI_COMPILE_FLAGS)
```

**Automatic Dependency Management**: The CMake script (`crs4/cpp/CMakeLists.txt`) automatically fetches and compiles the Cassandra C++ driver from source (v2.17.0) if it is not found on the system.

Key dependencies linked: `dali`, `cudart`, `cassandra` (C++ driver).

### Python Package (setup.py)

- Uses `setuptools` with custom `CMakeExtension` and `build_ext` command
- C++ compilation triggered via CMake when running `pip install .`
- Package name: `cassandra-dali-plugin`
- Python package: `crs4.cassandra_utils`
- Build requires: `setuptools>=64`, `wheel`, `cmake>=3.25.2`, `nvidia-dali-cuda130==1.53`
- Install requires: `cassandra-driver>=3.29.3`, `pandas>=3.0.1`, `tqdm>=4.67.3`

## Loading the Plugin

```python
import crs4.cassandra_utils
import nvidia.dali.plugin_manager as plugin_manager
import nvidia.dali.fn as fn
import pathlib

plugin_path = pathlib.Path(crs4.cassandra_utils.__path__[0])
plugin_path = plugin_path.parent.parent.joinpath("libcrs4cassandra.so")
plugin_manager.load_library(str(plugin_path))

# Use in pipeline (DALI operator name is fn.crs4.cassandra)
images, labels = fn.crs4.cassandra(
    name="Reader",
    cassandra_ips=["cassandra_host"],
    table="imagenet.train_data",
    label_col="label",
    label_type="int",
    data_col="data",
    id_col="img_id",
    source_uuids=train_uuids,
    prefetch_buffers=2,
)
```

## DALI Operator Parameters

### Core Parameters

| Parameter             | Description                                              | Default |
|-----------------------|----------------------------------------------------------|---------|
| `name`                | Reader name                                              | -       |
| `cassandra_ips`       | List of Cassandra IPs/hostnames                          | -       |
| `cassandra_port`      | Cassandra TCP port                                       | 9042    |
| `table`               | Data table name (e.g., `imagenet.data_train`)            | -       |
| `label_col`           | Label column name                                        | -       |
| `label_type`          | "int" (classification), "blob" (segmentation), or "none" | -       |
| `data_col`            | Data column name (BLOB)                                  | -       |
| `id_col`              | UUID column name                                         | -       |
| `source_uuids`        | Full list of UUIDs to retrieve                           | -       |
| `num_shards`          | Number of shards for distributed training                | 1       |
| `shard_id`            | Shard index for this process                             | 0       |
| `shuffle_every_epoch` | Shuffle UUIDs each epoch                                 | True    |
| `loop_forever`        | Loop dataset infinitely                                  | True    |

### Authentication

| Parameter               | Description                                              | Default |
|-------------------------|----------------------------------------------------------|---------|
| `username` / `password` | Cassandra auth credentials                               | None    |
| `use_ssl`               | Enable SSL                                               | False   |
| `ssl_certificate`       | Path to server public key                                | ""      |
| `ssl_own_certificate`   | Path to client public key                                | ""      |
| `ssl_own_key`           | Path to client private key                               | ""      |
| `ssl_own_key_pass`      | Password for client private key                          | ""      |
| `cloud_config`          | Astra-style dict `{'secure_connect_bundle': 'path.zip'}` | None    |

### Performance Tuning

| Parameter          | Description                                                   | Default |
|--------------------|---------------------------------------------------------------|---------|
| `prefetch_buffers` | Multi-buffering depth (hides latency)                         | 2       |
| `io_threads`       | Cassandra driver IO threads (limits TCP connections)          | 2       |
| `comm_threads`     | Communication handling threads                                | 2       |
| `copy_threads`     | Data copying threads                                          | 2       |
| `wait_threads`     | Wait handling threads                                         | 2       |
| `ooo`              | Out-of-order delivery (for high-latency/packet-loss networks) | False   |
| `slow_start`       | Prefetch dilution (request extra image every N)               | 0       |

## C++ Code Patterns

### Namespace

All code lives in `crs4` namespace.

### Reader Classes

Three reader implementations (all inherit from `dali::InputOperator<dali::CPUBackend>`):

1. **CassandraInteractive** - Synchronous batch prefetching, standard DALI pipeline
2. **CassandraSelffeed** - Self-feeding variant
3. **CassandraDecoupled** - Mini-batch decoupled for Triton inference server

### Class Pattern

```cpp
class CassandraInteractive : public dali::InputOperator<dali::CPUBackend> {
 public:
  explicit CassandraInteractive(const dali::OpSpec &spec);
  CassandraInteractive(const CassandraInteractive&) = delete;
  CassandraInteractive& operator=(const CassandraInteractive&) = delete;
  CassandraInteractive(CassandraInteractive&&) = delete;
  CassandraInteractive& operator=(CassandraInteractive&&) = delete;
  ~CassandraInteractive() override {
    if (batch_ldr != nullptr) {
      delete batch_ldr;
    }
  }
  // ...
};
```

### Header Guard Style

```cpp
#ifndef CRS4_CPP_CASSANDRA_DALI_INTERACTIVE_H_
#define CRS4_CPP_CASSANDRA_DALI_INTERACTIVE_H_
// ...
#endif  // CRS4_CPP_CASSANDRA_DALI_INTERACTIVE_H_
```

### Member Initialization

Prefer in-class initialization for pointers:

```cpp
BatchLoader* batch_ldr = nullptr;
size_t prefetch_buffers;
```

### ThreadPool

Custom `ThreadPool.h` (3rd party, from https://github.com/progschj/ThreadPool) used for IO parallelism. Three thread pools: `comm_pool`, `copy_pool`, `wait_pool`.

### Label Type Enum

```cpp
enum lab_type {lab_int, lab_img, lab_none};
```

## Python Code Patterns

### License Header

All source files include Apache 2.0 license header:

```python
# Copyright 2022 CRS4 (http://www.crs4.it/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
```

### Private Data Pattern

Examples use `private_data.template.py` as a template. Copy it to `private_data.py` and fill in:

```python
# examples/common/private_data.py
cass_conf = CassandraConf()
cass_conf.cassandra_ips = ["cassandra"]
cass_conf.cassandra_port = 9042
cass_conf.username = "cassandra"
cass_conf.password = "cassandra"
cass_conf.use_ssl = False
# ... SSL certs, cloud_config, etc.
```

All example scripts import from `private_data` (not the template).

### CassandraWriter Classes

- **CassandraWriter** - Base class
- **CassandraClassificationWriter** - For classification (`label_type="int"`)
- **CassandraSegmentationWriter** - For segmentation (`label_type="blob"`)

### CassandraConf

Simple dataclass-like class in `_cassandra_config.py` with fields: `username`, `password`, `cloud_config`, `cassandra_ips`, `cassandra_port`, `use_ssl`, `ssl_certificate`, `ssl_own_certificate`, `ssl_own_key`, `ssl_own_key_pass`.

## Data Model

Cassandra stores images/metadata in separate tables:

- **Metadata table**: Used for UUID selection via queries
- **Data table**: Stores actual binary data as BLOBs, accessed during training

The plugin reads exclusively from the data table during ML training. The `id_col` stores UUIDs, `label_col` stores labels (int or blob), and `data_col` stores the binary data.

### Typical Cassandra Schema

```sql
-- Example from examples/imagenette/create_tables.cql
CREATE KEYSPACE IF NOT EXISTS imagenette WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

CREATE TABLE IF NOT EXISTS imagenette.data_train (
    id uuid PRIMARY KEY,
    label int,
    data blob
);

CREATE TABLE IF NOT EXISTS imagenette.metadata_train (
    id uuid PRIMARY KEY,
    label int
);
```

The `metadata` table enables filtering by label during dataset preparation. The `data` table stores the actual image bytes.

## Docker Environment

| Component               | Version                                                           |
|-------------------------|-------------------------------------------------------------------|
| Base image              | NVIDIA PyTorch NGC Container (`nvcr.io/nvidia/pytorch:26.02-py3`) |
| Cassandra C++ driver    | 2.17.0 (Built automatically by CMake)                             |
| Cassandra Python driver | latest (via pip)                                                  |
| Spark                   | 3.5.x                                                             |
| DALI                    | Pre-installed in NGC container (1.53)                             |
| PyTorch Lightning       | 2.3.1                                                             |
| CUDA architectures      | 75;80;86;89;90                                                    |
| Default shell           | fish                                                              |

### Cassandra Container Access

The DALI container accesses Cassandra via SSH (keys in `varia/ssh/`):

```bash
ssh root@cassandra 'SSL_VALIDATE=false /opt/cassandra/bin/cqlsh --ssl -e "..."'
```

The `SSL_VALIDATE=false` flag is required for cqlsh over the Docker bridge network.

## Important Gotchas

1. **Cassandra startup delay**: Takes 1-2 minutes to become operational even after container starts. Test scripts include wait logic.

2. **SSL configuration**: `ssl_certificate`, `ssl_own_certificate`, `ssl_own_key` are **file paths**, not certificate content.

3. **Cloud config**: Astra-style config uses dict like `{'secure_connect_bundle': 'path-to-bundle.zip'}`.

4. **NVIDIA container runtime**: Requires `--ipc=host`, `SYS_ADMIN`, `NET_ADMIN` capabilities, and locked memory ulimits (`memlock=-1`, `stack=67108864`).

5. **Prefetch dilution** (`slow_start`): For high-latency/packet-loss networks, set `slow_start=N` to request an extra image every N normal requests, limiting initial burst.

6. **Out-of-order delivery** (`ooo=True`): For high-latency or lossy networks, returns images as soon as they arrive, potentially altering batch sequence and mixing batches.

7. **SSH access to Cassandra**: The DALI container connects to Cassandra via SSH on the Docker bridge network. The `SSL_VALIDATE=false` is mandatory for cqlsh in this setup.

8. **private_data.py**: Must be created from `private_data.template.py` before running examples. The template is copied to the container at build time.

9. **ThreadPool**: Third-party code from https://github.com/progschj/ThreadPool. Includes copyright notice but is freely usable.

## Performance Tuning for Long Fat Networks

See `docs/LFN.md` for detailed discussion. Key parameters for high-bandwidth, high-latency links:

- `prefetch_buffers`: 16 (deep prefetch queue)
- `io_threads`: 8 (many TCP connections)
- `comm_threads`: 1 (reduce contention)
- `copy_threads`: 4 (parallel copying)
- `ooo=True`: Handle packet loss gracefully
- `slow_start=4`: Dilute prefetch burst

## Citation

```bibtex
@misc{versaci2025hidinglatenciesnetworkbasedimage,
      title={Hiding Latencies in Network-Based Image Loading for Deep Learning},
      author={Francesco Versaci and Giovanni Busonera},
      year={2025},
      eprint={2503.22643},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2503.22643},
}
