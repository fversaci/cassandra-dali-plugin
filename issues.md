# Issues Found in Codebase

## 1. Hardcoded Timeout Values
**File:** `crs4/cpp/batch_loader.cc`
**Issue:** Connection and request timeouts are hardcoded (10000ms, 60000ms) without being configurable.
**Impact:** Inflexible for different network conditions. Should be parameters.

## 2. Third-Party Code Mixed with Project Code
**File:** `crs4/cpp/ThreadPool.h`
**Issue:** Third-party library (Jakob Progsch's ThreadPool) lives in the project source tree with no pinned version or clear separation.
**Impact:** Maintenance difficulty and unclear upgrade path. Should be tracked as a proper dependency.
