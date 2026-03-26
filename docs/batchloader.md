# BatchLoader Algorithm and Data Flow

This document explains the `BatchLoader` class in `crs4/cpp/batch_loader.{h,cc}`. `BatchLoader` is the low-level engine that connects Apache Cassandra query execution with DALI tensor storage and coordinates asynchronous loading, copying, and batch assembly.

## Purpose

`BatchLoader` is responsible for:

1. Connecting to Cassandra, either directly or through a cloud secure connection bundle.
2. Preparing a parameterized query used to fetch each sample by UUID.
3. Launching asynchronous Cassandra requests for all samples in a batch.
4. Copying Cassandra result buffers into DALI `TensorList` objects.
5. Supporting multiple in-flight batches through a buffer pool.
6. Optionally handling out-of-order completion.

In short, it turns a list of UUIDs into a DALI batch of features and labels.

---

## Main Data Structures

### Connection state

- `CassCluster* cluster`
- `CassSession* session`
- `const CassPrepared* prepared`
- `bool connected`

These objects manage the Cassandra connection lifecycle and the prepared statement used to fetch samples.

### Configuration

Key runtime parameters include:

- `table`
- `label_t`
- `label_col`
- `data_col`
- `id_col`
- `username`
- `password`
- `cassandra_ips`
- `cloud_config`
- `port`
- `use_ssl`
- SSL certificate/key paths
- `io_threads`
- `copy_threads`
- `wait_threads`
- `comm_threads`
- `prefetch_buffers`
- `ooo`

### Buffering and concurrency

`BatchLoader` uses several queues and vectors to manage multiple batches concurrently:

- `write_buf`: indices of free batch buffers
- `read_buf`: indices of buffers ready to be consumed
- `batch`: futures that resolve to `BatchImgLab`
- `comm_job`: futures for request dispatch jobs
- `copy_jobs`: futures for copy jobs, one vector per batch buffer
- `v_feats`: feature tensor lists
- `v_labs`: label tensor lists
- `bs`: batch sizes for each buffer
- `shapes`: per-sample feature sizes
- `lab_shapes`: per-sample label sizes for image labels
- `alloc_mtx` / `alloc_cv`: synchronization primitives used while allocating tensors
- `ooo_buf`: active buffers when out-of-order mode is enabled
- `ooo_in_bs`: per-buffer counters for out-of-order insertion

### Thread pools

`BatchLoader` uses three thread pools from `ThreadPool.h`:

- `comm_pool`: sends Cassandra queries
- `copy_pool`: copies bytes into DALI tensors
- `wait_pool`: waits for all copy operations to complete and assembles the final batch

---

## Type Aliases

The header defines:

- `lab_type`: one of `lab_int`, `lab_img`, `lab_none`
- `INT_LABEL_T`: alias for `int32_t`
- `BatchRawImage`: `dali::TensorList<dali::CPUBackend>`
- `BatchLabel`: `dali::TensorList<dali::CPUBackend>`
- `BatchImgLab`: `std::pair<BatchRawImage, BatchLabel>`

`BatchImgLab` is the final result returned by the loader.

---

## Initialization

### Constructor

The constructor stores all configuration values and initializes the multi-buffering structures.

If `label_type` is:

- `"int"`: labels are integer class ids
- `"blob"`: labels are binary blobs, typically segmentation masks
- `"none"`: no labels are fetched

The constructor also:

1. Resizes the batch-state vectors to `prefetch_buffers`.
2. Initializes `write_buf` with all available buffer indices.
3. Builds the Cassandra IP string from the vector of IPs, joined by commas.

### Connection is lazy

The Cassandra connection is not established in the constructor. Instead, `check_connection()` calls `connect()` on demand when the first batch is prefetched.

This allows object construction before the cluster is reachable and avoids connecting unless the loader is actually used.

---

## Connection Setup

### `connect()`

This method:

1. Configures the cluster connection.
2. Sets credentials and protocol version.
3. Applies SSL configuration if requested.
4. Opens the Cassandra session.
5. Prepares the query.
6. Creates the three thread pools.

#### Direct connection
If `cloud_config` is empty:

- `cass_cluster_set_contact_points(...)`
- `cass_cluster_set_port(...)`

#### Cloud connection
If `cloud_config` is set:

- `cass_cluster_set_cloud_secure_connection_bundle(...)`

#### Common options

- connect timeout: 10 seconds
- request timeout: 60 seconds
- protocol version: V4
- I/O thread count from `io_threads`
- queue size: 65536

### Query preparation

The prepared query is built as:

- `SELECT data_col FROM table WHERE id_col=?`
- or `SELECT label_col, data_col FROM table WHERE id_col=?` when labels are requested

The query is prepared once and reused for all samples.

---

## SSL Support

If `use_ssl` is enabled, `set_ssl()` configures the Cassandra SSL context.

Behavior:

- If no trusted certificate is provided, verification is disabled.
- If `ssl_certificate` is set, it is loaded as a trusted certificate.
- If both `ssl_own_certificate` and `ssl_own_key` are provided, client authentication is enabled.

Helper methods:

- `load_trusted_cert_file(...)`
- `load_own_cert_file(...)`
- `load_own_key_file(...)`

These read certificate/key files into memory and configure `CassSsl`.

---

## Batch Lifecycle

A batch goes through the following stages:

1. Reserve a buffer
2. Start asynchronous Cassandra queries
3. Receive query results
4. Allocate DALI tensors once all sample sizes are known
5. Copy data into the tensors
6. Wait for completion
7. Return the finished batch

---

## `prefetch_batch(...)`

This is the main entry point used by the DALI operator.

Steps:

1. Pop a free buffer index from `write_buf`.
2. Ensure the Cassandra connection exists via `check_connection()`.
3. Start batch transfers with `start_transfers(...)`.
4. Store the returned future in `batch[wb]`.
5. Push the buffer index into `read_buf`.

At this point the batch is in flight, and the caller can continue queuing additional work.

---

## `start_transfers(...)`

This method initializes the batch and launches the two main asynchronous flows:

- query dispatch in `comm_pool`
- completion tracking in `wait_pool`

Steps:

1. Store the batch size in `bs[wb]`.
2. Reserve space in `copy_jobs[wb]`.
3. Call `allocTens(wb)` to reset tensor lists and pre-allocate label storage when possible.
4. If `ooo` is enabled, register the buffer in `ooo_buf`.
5. Enqueue `keys2transfers(...)` in `comm_pool`.
6. Enqueue `wait4images(...)` in `wait_pool`.

The returned future becomes the representation of the whole batch.

---

## `allocTens(int wb)`

This method resets the buffer state and prepares the output tensor lists.

It:

- clears and resizes `shapes[wb]`
- resets `v_feats[wb]`
- resets `v_labs[wb]`
- marks both tensor lists as non-pinned
- pre-allocates label memory when labels are integers
- prepares `lab_shapes[wb]` when labels are images

Important detail:

- For integer labels, label storage can be allocated immediately.
- For image labels, the loader must first know each label size, so allocation is deferred until all samples are inspected.

---

## `keys2transfers(...)`

This runs in a communication thread and dispatches all Cassandra requests for the batch.

For each UUID:

1. Bind the UUID to the prepared statement.
2. Execute the query asynchronously using `cass_session_execute`.
3. Allocate a `futdata` structure containing:
   - the loader pointer
   - buffer index `wb`
   - sample index `i`
4. Register `wrap_enq` as the future callback.
5. Release the future immediately after setting the callback.

The Cassandra driver invokes the callback when the query completes.

---

## `wrap_enq(...)`

This is the callback registered on every Cassandra future.

It unwraps the `futdata` payload and then:

- calls `transfer2copy(...)` directly when `ooo` is disabled
- calls `ooo_enqueue(...)` when `ooo` is enabled

The `futdata` object is deleted inside the callback to avoid leaks.

---

## Out-of-Order Mode

When `ooo` is `true`, the loader allows completion order to differ from request order.

### Why it exists

Some Cassandra responses can take longer than others. In high-latency settings, waiting for strict request order can reduce throughput. Out-of-order mode allows completed results to be processed as soon as they arrive.

### `ooo_enqueue(...)`

This method selects the active out-of-order batch buffer and assigns the next slot in that buffer.

It:

1. Locks `ooo_buf_mtx`.
2. Uses `ooo_buf.front()` to select the active write buffer.
3. Assigns the next sample index from `ooo_in_bs[wb]`.
4. Increments `ooo_in_bs[wb]`.
5. If the buffer is now full, resets the counter and pops the buffer from `ooo_buf`.
6. Unlocks and calls `transfer2copy(...)`.

This means results are still copied into a single batch buffer, but the assignment of samples follows arrival order rather than request order.

---

## `transfer2copy(...)`

This method bridges the Cassandra result with the copy jobs and DALI tensor allocation logic.

Steps:

1. Extract the `CassResult` from the completed future.
2. Verify that a row is present.
3. Read the feature blob from `data_col`.
4. Record the feature size in `shapes[wb][i]`.
5. Depending on `label_t`, read the label and enqueue the appropriate copy function in `copy_pool`.
6. Store the returned copy future in `copy_jobs[wb]`.
7. If all sample sizes for the batch are known, allocate the feature tensor list and, if needed, the image-label tensor list.
8. Notify threads waiting on `alloc_cv`.

### Error handling

`transfer2copy(...)` throws if:

- the query failed
- the result is empty
- a Cassandra value cannot be decoded
- the label type is unknown

---

## Copy Functions

The copy functions are run by `copy_pool` threads.

### `copy_data_none(...)`

Used when no label is requested.

It:

1. Waits until the batch memory has been allocated.
2. Copies the image bytes into the feature tensor list.
3. Frees the Cassandra result.

### `copy_data_int(...)`

Used when labels are integers.

It:

1. Waits for allocation.
2. Copies the image bytes into the feature tensor list.
3. Copies the integer label into the label tensor list.
4. Frees the Cassandra result.

### `copy_data_img(...)`

Used when labels are blobs, usually image masks.

It:

1. Waits for allocation.
2. Copies the feature bytes.
3. Copies the label bytes.
4. Frees the Cassandra result.

### Synchronization detail

Each copy function waits on:

- `alloc_mtx[wb]`
- `alloc_cv[wb]`

until `copy_jobs[wb].size() == bs[wb]`.

That condition means the loader has seen enough samples to determine all shapes and allocate the output tensors.

---

## `wait4images(int wb)`

This method runs in `wait_pool` and converts the internal state into the final batch.

Steps:

1. Wait for `comm_job[wb]` to finish dispatching queries.
2. Wait until all copy jobs have been scheduled.
3. Call `.get()` on every copy future to propagate exceptions.
4. Clear `copy_jobs[wb]`.
5. Reset `comm_job[wb]`.
6. Move `v_feats[wb]` and `v_labs[wb]` into a `BatchImgLab`.
7. Return the pair.

This is the point at which a batch becomes fully materialized and ready for DALI.

---

## `blocking_get_batch()`

This is the consumer side of the pipeline.

Steps:

1. Pop the next ready buffer index from `read_buf`.
2. Wait for its `batch[rb]` future.
3. Push `rb` back into `write_buf`.
4. Return the finished `BatchImgLab`.

This call blocks until the batch is complete.

---

## `ignore_batch()`

Used during teardown.

If the loader is connected, it drains any in-flight batches by repeatedly calling `blocking_get_batch()` until `read_buf` is empty.

This ensures that all active futures are resolved before the Cassandra session and thread pools are destroyed.

---

## Resource Lifetime and Destruction

### Destructor

`~BatchLoader()`:

1. Calls `ignore_batch()` if connected.
2. Frees the Cassandra session.
3. Frees the Cassandra cluster.
4. Deletes the thread pools.

This prevents outstanding work from outliving the underlying Cassandra resources.

---

## Data Flow Summary

A typical batch follows this path:

1. `prefetch_batch(keys)`
2. `start_transfers(keys, wb)`
3. `keys2transfers(keys, wb)`
4. Cassandra completes individual queries
5. `wrap_enq(...)`
6. `transfer2copy(...)`
7. `copy_data_*(...)`
8. `wait4images(wb)`
9. `blocking_get_batch()`

---

## Important Invariants

A few invariants are relied upon by the implementation:

- `write_buf` contains only free buffers.
- `read_buf` contains only buffers with an active batch future.
- `copy_jobs[wb].size()` must equal `bs[wb]` before tensor allocation can be considered complete.
- All copy futures are collected by `wait4images()` before the batch is returned.
- A buffer index is returned to `write_buf` only after the batch has been consumed.
- In out-of-order mode, `ooo_buf` tracks which buffer is currently accepting arrivals.

Violating these assumptions would lead to deadlocks, data corruption, or use-after-free errors.

---

## Notes on Performance

The design is optimized for overlap:

- Cassandra requests are sent asynchronously.
- Result copying is parallelized.
- Completion waiting is separated from request dispatch.
- Multiple batch buffers allow the loader to keep the pipeline full.

The main trade-off is complexity in synchronization and memory ownership, especially around allocation timing and the lifetime of Cassandra result buffers.

---

## Limitations and Caveats

- The loader assumes each sample fits in memory as a contiguous blob.
- Tensor allocation for features is delayed until all sample sizes in the batch are known.
- The loader depends on Cassandra query results being available as binary blobs.
- Out-of-order mode adds scheduling flexibility but also more state to reason about.
- The code assumes successful Cassandra UUID binding and query execution; failures are propagated as exceptions.

---

## Relationship to the DALI Operators

`BatchLoader` is used by the DALI operators implemented in the C++ sources:

- `CassandraInteractive`
- `CassandraSelfFeed`
- `CassandraDecoupled`

Those operators handle DALI pipeline integration, while `BatchLoader` handles the actual Cassandra I/O and tensor assembly.

---

## Suggested Diagram

A useful mental model is:

