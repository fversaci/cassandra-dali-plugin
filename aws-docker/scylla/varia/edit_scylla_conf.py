# Copyright 2022 CRS4 (http://www.crs4.it/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# edit scylla.yaml to enable SSL
import yaml

yaml_fn = "/etc/scylla/scylla.yaml"
with open(yaml_fn, "r") as f:
    cass_conf = yaml.safe_load(f) or {}

# Ensure the necessary keys are present
cass_conf.setdefault("client_encryption_options", {})
cass_conf["read_request_timeout_in_ms"] = "60000"
cass_conf["write_request_timeout_in_ms"] = "60000"
cass_conf["commitlog_total_space_in_mb"] = "10000"
cass_conf["enable_cache"] = "false"
cass_conf["max_memory_for_unlimited_query_soft_limit"] = "1073741824"
cass_conf["max_memory_for_unlimited_query_hard_limit"] = "2147483648"
cass_conf["client_encryption_options"].update({
    "enabled": True,
    "certificate": "/tmp/scylla.crt",
    "keyfile": "/tmp/scylla.key"
})

with open(yaml_fn, "w") as f:
    yaml.dump(cass_conf, f, sort_keys=False)
