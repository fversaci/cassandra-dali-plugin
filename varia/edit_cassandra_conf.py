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

# edit cassandra.yaml to increase timeouts
# and enable SSL
import yaml

yaml_fn = "/cassandra/conf/cassandra.yaml"
with open(yaml_fn, "r") as f:
    cass_conf = yaml.safe_load(f)
    cass_conf["write_request_timeout"] = "20s"
    cass_conf["rpc_address"] = "0.0.0.0"
    cass_conf["broadcast_rpc_address"] = "127.0.0.1"
    cass_conf["client_encryption_options"]["enabled"] = True
    cass_conf["client_encryption_options"]["optional"] = True
    cass_conf["client_encryption_options"]["keystore"] = "/cassandra/conf/keystore"
    cass_conf["client_encryption_options"]["keystore_password"] = "cassandra"
    cass_conf.setdefault("sstable", {})["selected_format"] = "bti"
    cass_conf["concurrent_compactors"] = "4"
    cass_conf["compaction_throughput"] = "250MiB/s"
    cass_conf["storage_compatibility_mode"] = "NONE"
with open(yaml_fn, "w") as f:
    yaml.dump(cass_conf, f, sort_keys=False)


## # increase max direct memory -- Cassandra 4
## from pathlib import Path
## 
## jvm_fns = Path("/cassandra/conf/").glob("jvm*server.options")
## for fn in jvm_fns:
##     with open(fn, "a") as f:
##         f.write("-XX:MaxDirectMemorySize=64G")
