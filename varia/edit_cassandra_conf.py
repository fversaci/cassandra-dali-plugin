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

import os
import yaml
from pathlib import Path
import socket
from clize import run

# from IPython import embed

def yaml_conf():
    # change conf
    yaml_fn = "/cassandra/conf/cassandra.yaml"
    with open(yaml_fn, "r") as f:
        cass_conf = yaml.safe_load(f)
        cass_conf["cluster_name"] = "SslCluster"
        cass_conf["authenticator"] = "PasswordAuthenticator"
        cass_conf["authorizer"] = "CassandraAuthorizer"
        cass_conf["num_tokens"] = 256
        cass_conf["write_request_timeout"] = "20s"
        cass_conf["rpc_address"] = "0.0.0.0"
        cass_conf["broadcast_rpc_address"] = "127.0.0.1"
        # client-server encryption
        cass_conf["client_encryption_options"]["enabled"] = True
        cass_conf["client_encryption_options"]["optional"] = False
        cass_conf["client_encryption_options"]["keystore"] = "/certs/keystore.p12"
        cass_conf["client_encryption_options"]["keystore_password"] = "keystore"
        cass_conf["client_encryption_options"]["require_client_auth"] = True
        cass_conf["client_encryption_options"]["truststore"] = "/certs/truststore.p12"
        cass_conf["client_encryption_options"]["truststore_password"] = "truststore"
    with open(yaml_fn, "w") as f:
        yaml.dump(cass_conf, f, sort_keys=False)



def increase_mem():
    # increase max direct memory
    dir_mem_line = "-XX:MaxDirectMemorySize=64G\n"
    jvm_fns = Path("/cassandra/conf/").glob("jvm*server.options")
    for fn in jvm_fns:
        with open(fn, "r+") as f:
            lines = f.readlines()
            if dir_mem_line not in lines:
                f.write(dir_mem_line)


def do_config():
    yaml_conf()
    increase_mem()


# parse arguments
if __name__ == "__main__":
    run(do_config)
