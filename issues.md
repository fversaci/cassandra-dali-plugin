# Issues Found in Codebase

## 1. Missing Validation in CassandraWriter
**File:** `crs4/cassandra_utils/_cassandra_writer.py`
**Issue:** The base class does not validate that required parameters are provided before attempting to use them.
**Impact:** Cryptic errors later in the pipeline. Should validate in `__init__`.

## 2. Hardcoded Timeout Values
**File:** `crs4/cpp/batch_loader.cc`
**Issue:** Connection and request timeouts are hardcoded (10000ms, 60000ms) without being configurable.
**Impact:** Inflexible for different network conditions. Should be parameters.

## 3. Potential Division By Zero in Sharding
**File:** `crs4/cassandra_utils/_sharding.py`
**Function:** `get_shard`
**Issue:** No validation that `batch_size > 0` or `num_shards > 0`.
**Impact:** Division by zero errors. Should validate inputs.

## 4. Missing Null Check in CassandraSession
**File:** `crs4/cassandra_utils/_cassandra_session.py`
**Issue:** No validation that `cass_conf` is not None and has required attributes.
**Impact:** AttributeError on missing configuration. Should validate early.

## 5. Third-Party Code Mixed with Project Code
**File:** `crs4/cpp/ThreadPool.h`
**Issue:** Third-party library (Jakob Progsch's ThreadPool) lives in the project source tree with no pinned version or clear separation.
**Impact:** Maintenance difficulty and unclear upgrade path. Should be tracked as a proper dependency.

## 6. Potential Buffer Overflow in UUID Conversion
**File:** `crs4/cpp/cassandra_dali_interactive.cc`
**Function:** `prefetch_one`
**Issue:** Direct pointer arithmetic on `uint64_t*` without bounds checking assumes exactly 2 `uint64_t` values per tensor element.
**Impact:** If `uuids` tensor is malformed, could read out of bounds. Should validate tensor shape.

## 7. Missing Documentation for Complex Parameters
**File:** `crs4/cpp/cassandra_dali_interactive.cc`
**Issue:** The `ooo` (out-of-order) and `slow_start` parameters have minimal DALI_SCHEMA documentation.
**Impact:** Users may misuse these advanced parameters. Should add detailed documentation.

## 8. Missing Input Validation in CassandraSelfFeed
**File:** `crs4/cpp/cassandra_dali_selffeed.cc`
**Function:** `CassandraSelfFeed::CassandraSelfFeed`
**Issue:** No validation that `batch_size > 0`; emptiness of `source_uuids` is only checked after UUID conversion, not before.
**Impact:** Division by zero or misleading errors. Should validate early.
