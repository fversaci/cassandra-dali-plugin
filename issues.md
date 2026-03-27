# Issues Found in Codebase

## 1. Function Signature Mismatch in Training Scripts
**Files:** 
- `examples/splitfile/distrib_train_from_cassandra.py` (lines 341, 385, 389, 425)
- `examples/imagenette/distrib_train_from_cassandra.py` (lines 341, 385, 389, 425)

**Issue:** 
- `train()` defined with 5 parameters but called with 6 (includes `scaler`)
- `validate()` defined with 3 parameters but called with 4 (includes `scaler`)

**Impact:** TypeError at runtime when training starts.

## 2. Committed Private SSH Key
**File:** `varia/ssh/id_rsa`
**Issue:** Private SSH key committed to repository. Security risk.
**Impact:** Potential unauthorized access to infrastructure.

## 3. Hardcoded Python Version Path
**File:** `examples/triton/start-triton.sh` (line 2)
**Issue:** Hardcodes `/usr/local/lib/python3.12/dist-packages/`. Container may use different Python version.
**Impact:** Triton server may fail to load plugin if Python version differs.

## 4. Typo in Help Text
**File:** `examples/lightning/train_model.py` (line 45)
**Issue:** Typo "imagenette.data_vaò" instead of "imagenette.data_val".
**Impact:** Minor documentation error.

## 5. Duplicate Import
**File:** `examples/ade20k/extract_common.py` (lines 12, 15)
**Issue:** `import os` appears twice.
**Impact:** Code quality issue, no functional impact.

## 6. Incorrect Error Message
**File:** `crs4/cpp/batch_loader.cc` (line 168)
**Issue:** Error message in `load_own_key_file` says "Error loading SSL certificate" instead of "key".
**Impact:** Misleading error messages during SSL debugging.
