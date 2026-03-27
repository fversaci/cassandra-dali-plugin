# Issues Found in Codebase

## 1. NameError in Cassandra Writers
**File:** `crs4/cassandra_utils/_cassandra_classification_writer.py` (line 78), `crs4/cassandra_utils/_cassandra_segmentation_writer.py` (line 78)
**Issue:** Uses `concurrent.execute_concurrent_with_args` but module imported as `cassandra.concurrent`. Should be `cassandra.concurrent.execute_concurrent_with_args`.
**Impact:** Runtime error when calling `send_enqueued()`.

## 2. Function Signature Mismatch
**File:** `examples/splitfile/distrib_train_from_cassandra.py` (line 341, 385)
**Issue:** `train()` defined with 5 parameters `(train_loader, model, criterion, optimizer, epoch)` but called with 6 arguments including `scaler` at line 385.
**Impact:** TypeError at runtime when training starts.

## 3. Committed Private SSH Key
**File:** `varia/ssh/id_rsa`
**Issue:** Private SSH key committed to repository. Security risk.
**Impact:** Potential unauthorized access to infrastructure.

## 4. Hardcoded Python Version Path
**File:** `examples/triton/start-triton.sh` (line 2)
**Issue:** Hardcodes `/usr/local/lib/python3.12/dist-packages/`. Container may use different Python version.
**Impact:** Triton server may fail to load plugin if Python version differs.

## 5. Typo in Help Text
**File:** `examples/lightning/train_model.py` (line 45)
**Issue:** Typo "imagenette.data_vaò" instead of "imagenette.data_val".
**Impact:** Minor documentation error.

## 6. Duplicate Import
**File:** `examples/ade20k/extract_common.py` (lines 12, 15)
**Issue:** `import os` appears twice.
**Impact:** Code quality issue, no functional impact.

## 7. Incorrect Error Message
**File:** `crs4/cpp/batch_loader.cc` (line 168)
**Issue:** Error message in `load_own_key_file` says "Error loading SSL certificate" instead of "key".
**Impact:** Misleading error messages during SSL debugging.
