# Issues Found in Codebase

## 1. Committed Private SSH Key
**File:** `varia/ssh/id_rsa`
**Issue:** Private SSH key committed to repository. Security risk.
**Impact:** Potential unauthorized access to infrastructure.

## 2. Incorrect Error Message
**File:** `crs4/cpp/batch_loader.cc` (line 77)
**Issue:** Error message in `load_own_key_file` says "Error loading certificate file" instead of "key file".
**Impact:** Misleading error messages during SSL debugging.
