# BatchLoader

`BatchLoader` is the low-level loader that turns a list of Cassandra UUIDs into a DALI batch of features and labels.

## What it does

1. Opens a Cassandra connection on demand.
2. Prepares a parameterized query: `SELECT ... WHERE id_col=?`.
3. Sends async queries for all UUIDs in a batch.
4. Copies Cassandra result buffers into DALI `TensorList` objects.
5. Supports multiple in-flight batches.
6. Optionally accepts results out of order.

---

## Main state

### Connection
- `CassCluster* cluster`
- `CassSession* session`
- `const CassPrepared* prepared`
- `bool connected`

### Configuration
- Table and column names
- Cassandra credentials and IPs
- Cloud config and SSL settings
- Thread counts and `prefetch_buffers`
- Out-of-order mode (`ooo`)

### Buffering and concurrency
- `write_buf`: free batch buffers
- `read_buf`: ready-to-consume buffers
- `batch`: futures for completed batches
- `comm_job`: futures for query dispatch
- `copy_jobs`: futures for copy tasks
- `v_feats`, `v_labs`: feature and label tensor lists
- `bs`: batch size per buffer
- `shapes`, `lab_shapes`: per-sample sizes
- `ooo_buf`, `ooo_in_bs`: out-of-order bookkeeping

### Thread pools
- `comm_pool`: sends Cassandra queries
- `copy_pool`: copies bytes into DALI tensors
- `wait_pool`: waits for copies and assembles the batch

---

## Types

- `lab_type`: `lab_int`, `lab_img`, `lab_none`
- `INT_LABEL_T`: `int32_t`
- `BatchRawImage`: `dali::TensorList<dali::CPUBackend>`
- `BatchLabel`: `dali::TensorList<dali::CPUBackend>`
- `BatchImgLab`: `std::pair<BatchRawImage, BatchLabel>`

---

## Batch flow

A batch goes through these steps:

1. Reserve a free buffer.
2. Launch Cassandra requests.
3. Receive results.
4. Allocate DALI tensors once sample sizes are known.
5. Copy data into tensors.
6. Wait for all copy jobs.
7. Return the finished batch.

---

## Key methods

### `BatchLoader(...)`
Stores the configuration and initializes buffer state.  
The Cassandra connection is **lazy**: it is opened only when the first batch is prefetched.

### `connect()`
- Configures direct or cloud connection
- Sets credentials, protocol version, timeouts, and I/O threads
- Applies SSL if enabled
- Opens the session
- Prepares the query
- Creates the thread pools

### `prefetch_batch(const std::vector<CassUuid>& ks)`
- Takes a free buffer from `write_buf`
- Ensures the connection exists
- Starts batch transfers
- Moves the buffer to `read_buf`

### `start_transfers(const std::vector<CassUuid>& keys, int wb)`
- Stores batch size in `bs[wb]`
- Resets per-buffer state with `allocTens(wb)`
- Registers the buffer in `ooo_buf` if out-of-order mode is enabled
- Starts query dispatch in `comm_pool`
- Starts completion waiting in `wait_pool`

### `keys2transfers(...)`
Runs in the communication pool and submits one async Cassandra query per UUID:
- bind UUID to the prepared statement
- execute the query
- attach `wrap_enq` as callback

### `wrap_enq(...)`
Callback for each Cassandra future:
- calls `transfer2copy(...)` in normal mode
- calls `ooo_enqueue(...)` when out-of-order mode is enabled

### `transfer2copy(...)`
- Extracts the Cassandra result
- Reads feature bytes from `data_col`
- Reads label data if needed
- Enqueues the matching copy task
- Allocates DALI tensors once all sample sizes are known
- Notifies waiting copy threads

### Copy helpers
- `copy_data_none(...)`: copy features only
- `copy_data_int(...)`: copy features + integer label
- `copy_data_img(...)`: copy features + blob label

### `wait4images(int wb)`
- Waits for query dispatch to finish
- Waits for all copy jobs to be scheduled
- Propagates exceptions from copy tasks
- Moves the tensors into a `BatchImgLab`
- Returns the finished batch

### `blocking_get_batch()`
- Waits for the next ready batch
- Recycles the buffer back to `write_buf`
- Returns the batch data

### `ignore_batch()`
Drains any in-flight batches during shutdown so Cassandra and thread resources can be released cleanly.

---

## Out-of-order mode

When `ooo` is enabled, results are copied in arrival order instead of request order. This can improve throughput when query latency is uneven. The loader uses `ooo_buf` and `ooo_in_bs` to track the active batch buffer and the next insertion slot.

---

## Notes

- Features are assumed to be contiguous binary blobs.
- Feature tensor allocation is delayed until all sample sizes in a batch are known.
- Integer labels are allocated immediately; blob labels are allocated after sizes are discovered.
- The implementation is optimized to overlap Cassandra I/O, copying, and batch assembly.

---

## Relationship to the DALI operators

`BatchLoader` is used by:

- `CassandraInteractive`
- `CassandraSelfFeed`
- `CassandraDecoupled`

Those operators handle DALI pipeline integration; `BatchLoader` handles Cassandra I/O and tensor assembly.
