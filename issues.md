# Issues found

- **Major: split-file cross-validation logic is broken in `examples/splitfile/distrib_train_from_cassandra.py`.**
  The `crossval_index` option is documented, but `compute_split_index()` does not actually use it to pick the validation split. The branch also has fragile `exclude_index` handling that can crash when `exclude_index` is `None`. This makes the documented `--crossval-index` / `--exclude-index` feature unreliable.

- **Major: private SSH key committed to repository and embedded in Docker images.**
  `varia/ssh/id_rsa` contains an RSA private key tracked in the git repository. All three Dockerfiles (`Dockerfile.cassandra`, `Dockerfile.dali-cassandra`, `Dockerfile.dali-cassandra-triton`) copy this key into the resulting images. Anyone with repository access has the private key, which could allow unauthorized SSH access to any host that has the corresponding public key authorized.

- **Minor: Inconsistent use of `global_rank` vs `local_rank` in `examples/imagenette/distrib_train_from_cassandra.py`.**
  In `main()`, `local_rank` is used for `shard_id` in `create_dali_pipeline` and `device_id`, but `global_rank` is defined at the top level. In `train()` and `validate()`, `local_rank` is used for logging, but `world_size` is used for `reduce_tensor`. This might be intended, but it's confusing and potentially incorrect if `local_rank` is not the correct rank for logging or sharding.

- **Minor: Inconsistent use of `global_rank` vs `local_rank` in `examples/imagenette/distrib_train_from_file.py`.**
  Similar to `distrib_train_from_cassandra.py`, there is potential confusion between `local_rank` and `global_rank` for sharding and logging.
