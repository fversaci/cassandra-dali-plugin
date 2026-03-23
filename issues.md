# Issues found

- **Major: `crs4/cassandra_utils/_cassandra_segmentation_writer.py` uses `BatchStatement` without importing it.**  
  `save_item()` calls `BatchStatement()` but that symbol is never imported, so saving items will raise `NameError`.

- **Major: split-file cross-validation logic is broken in `examples/splitfile/distrib_train_from_cassandra.py`.**  
  The `crossval_index` option is documented, but `compute_split_index()` does not actually use it to pick the validation split. The branch also has fragile `exclude_index` handling that can crash when `exclude_index` is `None`. This makes the documented `--crossval-index` / `--exclude-index` feature unreliable.

- **Major: the Corel-5k example documentation and implementation do not agree on how data is serialized.**  
  `examples/corel5k/README.md` says images are stored as JPEG blobs and labels as NPY blobs, but the current writer path in `examples/corel5k/extract_common.py` applies the same `get_data(img_format)` logic to both image and label files. That means the example only works cleanly in the `UNCHANGED` path, not for the documented JPEG+NPY setup.

- **Major: Triton startup script hardcodes a likely wrong plugin path.**  
  `examples/triton/start-triton.sh` points Triton to `/usr/local/lib/python3.12/dist-packages/libcrs4cassandra.so`, which is inconsistent with the container setup and the Python version used in the Dockerfiles. This is likely to prevent the backend plugin from loading.

- **Major: the decoupled Triton stress model config has an output-type mismatch.**  
  `examples/triton/models/dali_cassandra_decoupled_stress/config.pbtxt` declares `DALI_OUTPUT_0` as `TYPE_FP32`, but the corresponding pipeline is a raw-byte / first-byte stress test and does not obviously produce FP32 output. The config and implementation look inconsistent.

- **Major: `examples/splitfile/README.md` has malformed markdown.**  
  The first code block is not properly closed before the `## Create a split file` section, so the rest of the document will render incorrectly.

- **Security issue: an SSH private key is committed to the repository and copied into container images.**  
  `varia/ssh/id_rsa` is present in the repo, and the Dockerfiles copy `varia/ssh/` into the images. This exposes a secret key in source control and in built images.
