# Issues found

- **Major: split-file cross-validation logic is broken in `examples/splitfile/distrib_train_from_cassandra.py`.**  
  The `crossval_index` option is documented, but `compute_split_index()` does not actually use it to pick the validation split. The branch also has fragile `exclude_index` handling that can crash when `exclude_index` is `None`. This makes the documented `--crossval-index` / `--exclude-index` feature unreliable.

- **Major: the decoupled Triton stress model config has an output-type mismatch.**  
  `examples/triton/models/dali_cassandra_decoupled_stress/config.pbtxt` declares `DALI_OUTPUT_0` as `TYPE_FP32`, but the corresponding pipeline is a raw-byte / first-byte stress test and does not obviously produce FP32 output. The config and implementation look inconsistent.
