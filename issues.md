# Issues Found in Codebase

## 1. Committed Private SSH Key
**File:** `varia/ssh/id_rsa`
**Issue:** Private SSH key committed to repository. Security risk.
**Impact:** Potential unauthorized access to infrastructure.

## 2. Use-After-Free in Cassandra Asynchronous Fetch
**File:** `crs4/cpp/batch_loader.cc`
**Function:** `transfer2copy`
**Issue:** `cass_future_free(query_future)` is called immediately after enqueuing a copy job that relies on `result` (derived from that future). The Cassandra C driver invalidates the result memory when the future is freed.
**Impact:** Race condition leading to segmentation faults or data corruption during batch loading, as the worker thread may access freed memory.

## 3. Incorrect Padding Logic in Sharding
**File:** `crs4/cassandra_utils/_sharding.py`
**Function:** `uuids_as_tensors`
**Issue:** The calculation `bs - len(uuids) % bs` adds a full batch of padding when `len(uuids)` is perfectly divisible by `bs` (remainder is 0).
**Impact:** Unnecessary memory allocation and processing of an extra dummy batch at the end of the dataset.

## 4. Unreliable Resource Cleanup in CassandraSession
**File:** `crs4/cassandra_utils/_cassandra_session.py`
**Class:** `CassandraSession`
**Issue:** Relies on `__del__` to call `self.cluster.shutdown()`. In Python, `__del__` invocation is non-deterministic and can be skipped during interpreter shutdown.
**Impact:** Resource leaks or errors if the session is not explicitly managed. Should implement a context manager (`__enter__`, `__exit__`).

## 5. Deprecated SSL Protocol Constant
**File:** `crs4/cassandra_utils/_cassandra_session.py`
**Function:** `CassandraSession.__init__`
**Issue:** Uses `ssl.PROTOCOL_TLS`, which is deprecated since Python 3.10.
**Impact:** Runtime deprecation warnings. Should use `ssl.PROTOCOL_TLS_CLIENT`.

## 6. Missing Error Handling for File Reads
**File:** `crs4/cpp/batch_loader.cc`
**Functions:** `load_own_cert_file`, `load_own_key_file`, `load_trusted_cert_file`
**Issue:** Calls to `fread` do not check the return value (number of items read).
**Impact:** If a partial read occurs, the SSL handshake will fail with invalid data, but the error message will be misleading.
