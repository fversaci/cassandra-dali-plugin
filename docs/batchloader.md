# BatchLoader Algorithm and Data Flow

This document details the internal algorithm, threading model, and data flow of the `BatchLoader` class, which serves as the core component for bridging Apache Cassandra with the NVIDIA DALI pipeline.

## Overview

The `BatchLoader` is responsible for:
1.  Managing connections to a Cassandra cluster.
2.  Asynchronously fetching image data and labels based on UUIDs.
3.  Managing memory allocation for DALI Tensors.
4.  Copying data from Cassandra driver buffers into DALI-readable memory.
5.  Implementing multi-buffering to overlap I/O with computation.

## Architecture Components

### Thread Pools
To achieve high parallelism, `BatchLoader` utilizes three distinct thread pools:
*   **`comm_pool` (Communication Threads):** Handles the dispatch of Cassandra queries. It iterates through the list of UUIDs and sends asynchronous requests to the Cassandra cluster.
*   **`copy_pool` (Copy Threads):** Handles the memory copy operations (`memcpy`) from the Cassandra driver's memory into the DALI Tensor memory. This is CPU-bound work.
*   **`wait_pool` (Wait Threads):** Handles the synchronization of batch completion. It waits for all copy operations of a specific batch to finish before marking the batch as ready.

### Buffer Management
The loader uses a multi-buffering strategy (double buffering or more) defined by `prefetch_buffers`.
*   **`write_buf`**: A queue of buffer indices that are free to be written to (empty buffers).
*   **`read_buf`**: A queue of buffer indices that are filled and ready to be consumed.
*   **`v_feats` / `v_labs`**: Vectors of TensorLists holding the actual image and label data.

### Synchronization Primitives
*   **`alloc_mtx` / `alloc_cv`**: A mutex and condition variable pair (one per buffer) used to synchronize the allocation of tensor memory. Copy threads must wait until the main thread determines the total size of the batch and allocates the memory block.

## Data Flow Algorithm

The lifecycle of a batch involves distinct stages: Prefetch, Transfer, Allocation, Copy, and Retrieval.

### 1. Prefetching (`prefetch_batch`)
When the DALI operator requests data for the next batch:
1.  A buffer index `wb` is popped from `write_buf`.
2.  `start_transfers` is called to initiate the asynchronous workflow.
3.  A future representing the final batch is stored in `batch[wb]`.
4.  The buffer index `wb` is pushed to `read_buf`.

### 2. Initiating Transfers (`start_transfers`)
1.  The batch size is recorded.
2.  `allocTens` initializes empty TensorLists and clears shape vectors.
3.  If **Out-of-Order (OOO)** execution is disabled:
    *   `keys2transfers` is enqueued into `comm_pool`.
4.  If **OOO** is enabled:
    *   The buffer index is pushed to `ooo_buf`.
    *   `keys2transfers` is enqueued.
5.  A task `wait4images` is enqueued into `wait_pool` to monitor completion.

### 3. Dispatching Queries (`keys2transfers`)
This function runs in a `comm_pool` thread:
1.  Iterates through the provided vector of UUIDs.
2.  Binds the UUID to the prepared statement.
3.  Executes `cass_session_execute` (asynchronous Cassandra query).
4.  Sets a callback `wrap_enq` on the future. This callback triggers when the Cassandra driver receives a response.

### 4. Handling Query Results (`wrap_enq` / `transfer2copy`)
When the Cassandra driver completes a query:
1.  **OOO Disabled**: `transfer2copy` is called directly.
2.  **OOO Enabled**: `ooo_enqueue` manages the order of completion. It tracks how many items have returned for the current batch. Once a batch is fully populated in the OOO buffer logic, it triggers `transfer2copy`.

### 5. Scheduling Copy and Allocation (`transfer2copy`)
This is a critical step that bridges the network response and memory management:
1.  **Extract Data**: The raw bytes (image/label) are extracted from the Cassandra result.
2.  **Record Size**: The size of the current sample is stored in `shapes[wb][i]`.
3.  **Enqueue Copy**: A copy job is created and enqueued into `copy_pool`.
4.  **Allocation Logic**: The thread locks `alloc_mtx`.
    *   It pushes the copy job future into `copy_jobs[wb]`.
    *   **Trigger**: If `copy_jobs[wb].size() == batch_size`, it means all query results for this batch have arrived and their sizes are known.
    *   **Action**: The thread calculates the total `TensorListShape` and calls `v_feats[wb].Resize(...)`. This allocates the contiguous memory block for the batch.
5.  **Notify**: `alloc_cv` is notified to wake up any copy threads waiting for memory allocation.

### 6. Copying Data (`copy_data_none`, `copy_data_int`, `copy_data_img`)
These functions run in `copy_pool` threads:
1.  **Wait**: Acquires `alloc_mtx` and waits on `alloc_cv` until `copy_jobs[wb].size() == batch_size`. This ensures the memory has been allocated (Step 5).
2.  **Copy**: Performs `std::memcpy` from the Cassandra result pointer to the specific offset in the DALI TensorList (`v_feats[wb]`).
3.  **Cleanup**: Frees the Cassandra result memory.

### 7. Waiting for Completion (`wait4images`)
This function runs in the `wait_pool`:
1.  Waits for the `comm_job` (dispatching queries) to finish.
2.  Waits for all `copy_jobs` (copying data) to finish.
3.  Moves the populated `v_feats` and `v_labs` into a pair and returns it.

### 8. Retrieving the Batch (`blocking_get_batch`)
Called by the DALI operator to get the processed data:
1.  Pops a buffer index `rb` from `read_buf`.
2.  Calls `.get()` on the `batch[rb]` future. This blocks until `wait4images` completes.
3.  Pushes the buffer index `rb` back to `write_buf` to be reused.
4.  Returns the data.

## Out-of-Order (OOO) Execution

The OOO mode is an optimization for high-latency or high-concurrency scenarios.
*   **Standard Mode**: Results are processed strictly in the order they were requested (or at least, associated with specific buffer slots immediately).
*   **OOO Mode**: The `ooo_buf` queue acts as a holding area. As Cassandra results arrive (which may be out of order relative to the request stream), `ooo_enqueue` assigns them to the current active batch buffer. This allows the system to fill buffers dynamically as data arrives, potentially reducing idle time if specific queries take longer than others.

## Summary Diagram

