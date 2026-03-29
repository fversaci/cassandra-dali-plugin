# Issues Found in Codebase

## 1. Incorrect Padding Logic in Sharding
**File:** `crs4/cassandra_utils/_sharding.py`
**Function:** `uuids_as_tensors`
**Issue:** The calculation `bs - len(uuids) % bs` adds a full extra batch of padding when `len(uuids)` is perfectly divisible by `bs` (remainder is 0).
**Impact:** Unnecessary memory allocation and processing of an extra dummy batch at the end of the dataset.

## 2. Missing Validation in CassandraWriter
**File:** `crs4/cassandra_utils/_cassandra_writer.py`
**Issue:** The base class does not validate that required parameters are provided before attempting to use them.
**Impact:** Cryptic errors later in the pipeline. Should validate in `__init__`.

## 3. Hardcoded Timeout Values
**File:** `crs4/cpp/batch_loader.cc`
**Issue:** Connection and request timeouts are hardcoded (10000ms, 60000ms) without being configurable.
**Impact:** Inflexible for different network conditions. Should be parameters.

## 4. Potential Division By Zero in Sharding
**File:** `crs4/cassandra_utils/_sharding.py`
**Function:** `get_shard`
**Issue:** No validation that `batch_size > 0` or `num_shards > 0`.
**Impact:** Division by zero errors. Should validate inputs.

## 5. Missing Null Check in CassandraSession
**File:** `crs4/cassandra_utils/_cassandra_session.py`
**Issue:** No validation that `cass_conf` is not None and has required attributes.
**Impact:** AttributeError on missing configuration. Should validate early.

## 6. Third-Party Code Mixed with Project Code
**File:** `crs4/cpp/ThreadPool.h`
**Issue:** Third-party library (Jakob Progsch's ThreadPool) lives in the project source tree with no pinned version or clear separation.
**Impact:** Maintenance difficulty and unclear upgrade path. Should be tracked as a proper dependency.

## 7. Potential Buffer Overflow in UUID Conversion
**File:** `crs4/cpp/cassandra_dali_interactive.cc`
**Function:** `prefetch_one`
**Issue:** Direct pointer arithmetic on `uint64_t*` without bounds checking assumes exactly 2 `uint64_t` values per tensor element.
**Impact:** If `uuids` tensor is malformed, could read out of bounds. Should validate tensor shape.

## 8. Race Condition in Batch Shape Assignment
**File:** `crs4/cpp/batch_loader.cc`
**Function:** `transfer2copy`
**Issue:** The assignment `shapes[wb][i] = sz` (and `lab_shapes[wb][i] = l_sz` for segmentation) occurs outside the `alloc_mtx[wb]` mutex lock. Multiple threads can write to different indices of the same vector concurrently, and the final read by the last thread (to allocate tensors) may not see all writes due to lack of synchronization.
**Impact:** Data race (Undefined Behavior). The thread allocating the tensor may read uninitialized/incorrect shape values, leading to tensor allocation failures or memory corruption.

## 9. Missing Documentation for Complex Parameters
**File:** `crs4/cpp/cassandra_dali_interactive.cc`
**Issue:** The `ooo` (out-of-order) and `slow_start` parameters have minimal DALI_SCHEMA documentation.
**Impact:** Users may misuse these advanced parameters. Should add detailed documentation.

## 10. Missing Input Validation in CassandraSelfFeed
**File:** `crs4/cpp/cassandra_dali_selffeed.cc`
**Function:** `CassandraSelfFeed::CassandraSelfFeed`
**Issue:** No validation that `batch_size > 0`; emptiness of `source_uuids` is only checked after UUID conversion, not before.
**Impact:** Division by zero or misleading errors. Should validate early.
