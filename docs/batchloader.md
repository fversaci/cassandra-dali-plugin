# BatchLoader

`BatchLoader` is a C++ class that provides high-performance, asynchronous data loading from a Cassandra database into NVIDIA DALI tensors. It bridges the gap between Cassandra's distributed storage and DALI's GPU-accelerated data pipeline.

## Overview

`BatchLoader` manages the entire lifecycle of reading a batch of data:
1.  **Connection Management**: Handles Cassandra cluster connections, authentication, and SSL/TLS encryption, including support for cloud deployments (e.g., DataStax Astra).
2.  **Asynchronous I/O**: Uses non-blocking Cassandra queries and thread pools to parallelize network requests and data copying.
3.  **Memory Management**: Allocates and populates DALI `TensorList` objects on the CPU, ready for transfer to the GPU.
4.  **Multi-buffering**: Supports multiple in-flight batches to ensure the DALI pipeline is never starved of data.

---

## Configuration

The `BatchLoader` is configured via its constructor arguments:

| Argument               | Type                       | Description                                                  |
|:-----------------------|:---------------------------|:-------------------------------------------------------------|
| `table`                | `std::string`              | Fully qualified table name (e.g., `keyspace.table_name`).    |
| `label_type`           | `std::string`              | Type of label: `"int"`, `"blob"`, or `"none"`.               |
| `label_col`            | `std::string`              | Column name for the label/mask.                              |
| `data_col`             | `std::string`              | Column name for the feature data (blob).                     |
| `id_col`               | `std::string`              | Column name for the primary key (UUID).                      |
| `username`, `password` | `std::string`              | Cassandra authentication credentials.                        |
| `cassandra_ips`        | `std::vector<std::string>` | List of contact points for the cluster.                      |
| `port`                 | `int`                      | Cassandra port (default: 9042).                              |
| `cloud_config`         | `std::string`              | Path to the secure connection bundle for AstraDB (optional). |
| `use_ssl`              | `bool`                     | Enable SSL/TLS encryption.                                   |
| `ssl_certificate`      | `std::string`              | Path to the trusted server certificate.                      |
| `ssl_own_certificate`  | `std::string`              | Path to the client certificate (mutual TLS).                 |
| `ssl_own_key`          | `std::string`              | Path to the client private key.                              |
| `ssl_own_key_pass`     | `std::string`              | Password for the client private key.                         |
| `io_threads`           | `size_t`                   | Number of I/O threads for the Cassandra driver.              |
| `prefetch_buffers`     | `size_t`                   | Number of batch buffers to cycle through.                    |
| `copy_threads`         | `size_t`                   | Number of threads for copying data into DALI tensors.        |
| `wait_threads`         | `size_t`                   | Number of threads for waiting on batch completion.           |
| `comm_threads`         | `size_t`                   | Number of threads for dispatching Cassandra queries.         |
| `ooo`                  | `bool`                     | Enable out-of-order processing of results.                   |

---

## Types

- **`lab_type`**: Enum for label types (`lab_int`, `lab_img`, `lab_none`).
- **`INT_LABEL_T`**: Type alias for integer labels (`int32_t`).
- **`BatchRawImage`**: `dali::TensorList<dali::CPUBackend>` for feature data.
- **`BatchLabel`**: `dali::TensorList<dali::CPUBackend>` for label data.
- **`BatchImgLab`**: `std::pair<BatchRawImage, BatchLabel>`, the return type for a finished batch.

---

## Internal State

### Connection
- `cluster`, `session`, `prepared`: Handles for the Cassandra driver connection and prepared statement.
- `connected`: Boolean flag to prevent re-connection.

### Configuration
- Stores table/column names, credentials, SSL settings, and thread counts.

### Buffering & Concurrency
- **Buffers**:
  - `write_buf`: Queue of buffer indices available for new prefetches.
  - `read_buf`: Queue of buffer indices ready for consumption.
  - `ooo_buf`: Queue for active buffers in out-of-order mode.
- **Tensors**:
  - `v_feats`, `v_labs`: Vectors of DALI TensorLists (one per buffer).
  - `shapes`, `lab_shapes`: Stores the dimensions of each sample in the batch.
- **Synchronization**:
  - `alloc_mtx`, `alloc_cv`: Mutex and condition variable to synchronize tensor allocation with data copying.
  - `ooo_buf_mtx`: Protects the out-of-order buffer queue.

### Thread Pools
- `comm_pool`: Dispatches Cassandra queries.
- `copy_pool`: Copies raw bytes from Cassandra results into DALI tensors.
- `wait_pool`: Waits for all copy tasks to complete and assembles the final batch.

---

## Batch Lifecycle

The processing of a batch follows a strict pipeline managed by the class:

1.  **Prefetch Request**:
    `prefetch_batch(keys)` is called with a list of UUIDs.
    - A buffer index `wb` is popped from `write_buf`.
    - `check_connection()` ensures the database is connected.
    - `start_transfers(keys, wb)` is invoked.

2.  **Transfer Initialization** (`start_transfers`):
    - The batch size is stored.
    - `allocTens(wb)` resets the tensor shapes and clears previous data.
    - If Out-of-Order (OOO) mode is active, the buffer is pushed to `ooo_buf`.
    - `keys2transfers` is enqueued in `comm_pool`.
    - `wait4images` is enqueued in `wait_pool` to handle completion.
    - The buffer index `wb` is pushed to `read_buf`.

3.  **Query Dispatch** (`keys2transfers`):
    - Iterates through UUIDs.
    - Binds each UUID to the prepared statement.
    - Executes the query asynchronously with a callback (`wrap_enq`).

4.  **Result Handling** (`wrap_enq` -> `transfer2copy`):
    - The callback is triggered when a Cassandra query completes.
    - **OOO Mode**: `ooo_enqueue` determines the correct slot in the batch.
    - **Standard Mode**: Uses the pre-determined index.
    - `transfer2copy` extracts the raw bytes and label.
    - A copy task is enqueued in `copy_pool`.
    - **Allocation Logic**: When the number of scheduled copy jobs equals the batch size, the DALI tensors are allocated based on the gathered `shapes`. `alloc_cv` is notified.

5.  **Data Copy** (`copy_data_*`):
    - Waits on `alloc_cv` until tensors are allocated.
    - Performs `memcpy` from the Cassandra result to the DALI tensor.
    - Frees the Cassandra result memory.

6.  **Completion** (`wait4images`):
    - Waits for `comm_job` (query dispatch) to finish.
    - Waits for all `copy_jobs` to finish.
    - Moves the tensors into a `BatchImgLab` pair and returns it.

7.  **Consumption** (`blocking_get_batch`):
    - Pops a buffer index from `read_buf`.
    - Calls `.get()` on the future to retrieve the `BatchImgLab`.
    - Pushes the buffer index back to `write_buf` for reuse.

---

## Key Methods

### `BatchLoader(...)`
Initializes configuration parameters and internal buffer queues. The connection itself is **lazy** and deferred until the first `prefetch_batch` call.

### `connect()`
Establishes the connection to the Cassandra cluster.
- Configures contact points or cloud secure bundle.
- Sets authentication and SSL/TLS settings.
- Prepares the parameterized `SELECT` query.
- Initializes the `comm`, `copy`, and `wait` thread pools.

### `prefetch_batch(const std::vector<CassUuid>& keys)`
Initiates the asynchronous load for a batch of UUIDs. It manages the transition of buffer indices from `write_buf` to `read_buf`.

### `blocking_get_batch()`
A blocking call that returns the next available `BatchImgLab`. It handles the synchronization with the background threads and recycles the buffer.

### `ignore_batch()`
A cleanup utility that drains all in-flight batches. This is essential for the destructor to ensure all Cassandra futures are resolved before freeing the session object.

---

## Out-of-Order Mode

When `ooo` is enabled, the loader processes results as they arrive from the network, rather than strictly following the order of the input UUID vector. This can reduce latency when query response times vary significantly.

- **Mechanism**: Uses `ooo_buf` to track the current active batch buffer. `ooo_enqueue` atomically determines the next available slot (`ooo_in_bs`) within that buffer.
- **Synchronization**: Protected by `ooo_buf_mtx` to ensure thread-safe index assignment.

---

## SSL/TLS Support

`BatchLoader` supports secure connections:
- **Server Verification**: If `ssl_certificate` is provided, the server's certificate is validated against it.
- **Client Authentication**: If `ssl_own_certificate` and `ssl_own_key` are provided, the driver performs mutual TLS authentication.
- **Implementation**: Uses `cass_ssl_set_*` functions to configure the SSL context before connecting.

---

## Relationship to DALI Operators

This class is the backend engine for the DALI Cassandra operators:
- **`CassandraInteractive`**: Used for interactive inference workloads.
- **`CassandraDecoupled`**: Used for decoupled training/inference pipelines.
- **`CassandraSelfFeed`**: Used for pipelines where the reader feeds itself.

The operators manage the DALI pipeline integration (inputs/outputs), while `BatchLoader` manages the heavy lifting of database I/O and memory management.
