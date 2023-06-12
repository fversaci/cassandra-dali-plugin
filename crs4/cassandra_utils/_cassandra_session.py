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

# pip3 install cassandra-driver
import cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import TokenAwarePolicy, DCAwareRoundRobinPolicy
from cassandra.cluster import ExecutionProfile
import pandas as pd
import ssl

def pandas_factory(colnames, rows):
    return pd.DataFrame(rows, columns=colnames)

class CassandraSession:
    def __init__(self, cass_conf):
        self.cass_conf = cass_conf
        # read parameters
        auth_prov = PlainTextAuthProvider(
            username=cass_conf.username, password=cass_conf.password
        )
        cassandra_ips = cass_conf.cassandra_ips
        cloud_config = cass_conf.cloud_config
        port = cass_conf.cassandra_port
        use_ssl = (cass_conf.use_ssl,)
        # set profiles
        prof_dict = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory=cassandra.query.dict_factory,
        )
        prof_tuple = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory=cassandra.query.tuple_factory,
        )
        prof_pandas = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory=pandas_factory,
        )
        profs = {"dict": prof_dict, "tuple": prof_tuple, "pandas": prof_pandas}
        # init cluster
        if cloud_config:
            self.cluster = Cluster(
                cloud=cloud_config,
                execution_profiles=profs,
                protocol_version=4,
                auth_provider=auth_prov,
            )
        else:
            if use_ssl:
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
            else:
                ssl_context = None
            self.cluster = Cluster(
                contact_points=cassandra_ips,
                execution_profiles=profs,
                protocol_version=4,
                auth_provider=auth_prov,
                port=port,
                ssl_context=ssl_context,
            )
        self.cluster.connect_timeout = 10  # seconds
        # start session
        self.sess = self.cluster.connect()

    def __del__(self):
        self.cluster.shutdown()
