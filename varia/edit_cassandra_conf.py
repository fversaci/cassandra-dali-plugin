# edit cassandra.yaml to increase timeouts
# and enable SSL
import yaml

yaml_fn = "/cassandra/conf/cassandra.yaml"
with open(yaml_fn, "r") as f:
    cass_conf = yaml.safe_load(f)
    cass_conf["write_request_timeout_in_ms"] = 20000
    cass_conf["rpc_address"] = "0.0.0.0"
    cass_conf["broadcast_rpc_address"] = "127.0.0.1"
    cass_conf["client_encryption_options"]["enabled"] = True
    cass_conf["client_encryption_options"]["optional"] = False
    cass_conf["client_encryption_options"]["keystore"] = "/cassandra/conf/keystore"
with open(yaml_fn, "w") as f:
    yaml.dump(cass_conf, f, sort_keys=False)


# increase max direct memory
from pathlib import Path

jvm_fns = Path("/cassandra/conf/").glob("jvm*server.options")
for fn in jvm_fns:
    with open(fn, "a") as f:
        f.write("-XX:MaxDirectMemorySize=32G")
