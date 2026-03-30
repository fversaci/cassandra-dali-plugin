# Issues Found in Codebase

## 1. Hardcoded Timeout Values
**File:** `crs4/cpp/batch_loader.cc`
**Issue:** Connection and request timeouts are hardcoded (10000ms, 60000ms) without being configurable.
**Impact:** Inflexible for different network conditions. Should be parameters.

## 2. Third-Party Code Mixed with Project Code
**File:** `crs4/cpp/ThreadPool.h`
**Issue:** Third-party library (Jakob Progsch's ThreadPool) lives in the project source tree with no pinned version or clear separation.
**Impact:** Maintenance difficulty and unclear upgrade path. Should be tracked as a proper dependency.

## 3. Potential Buffer Overflow in UUID Conversion
**File:** `crs4/cpp/cassandra_dali_interactive.cc`
**Function:** `prefetch_one`
**Issue:** Direct pointer arithmetic on `uint64_t*` without bounds checking assumes exactly 2 `uint64_t` values per tensor element.
**Impact:** If `uuids` tensor is malformed, could read out of bounds. Should validate tensor shape.

## 4. Missing Documentation for Complex Parameters
**File:** `crs4/cpp/cassandra_dali_interactive.cc`
**Issue:** The `ooo` (out-of-order) and `slow_start` parameters have minimal DALI_SCHEMA documentation.
**Impact:** Users may misuse these advanced parameters. Should add detailed documentation.

## 5. Missing Input Validation in CassandraSelfFeed
**File:** `crs4/cpp/cassandra_dali_selffeed.cc`
**Function:** `CassandraSelfFeed::CassandraSelfFeed`
**Issue:** No validation that `batch_size > 0`; emptiness of `source_uuids` is only checked after UUID conversion, not before.
**Impact:** Division by zero or misleading errors. Should validate early.
