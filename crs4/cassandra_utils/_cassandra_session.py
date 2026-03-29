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
import ssl


class CassandraSession:
    def __init__(self, cass_conf):
        # read parameters
        auth_prov = PlainTextAuthProvider(
            username=cass_conf.username, password=cass_conf.password
        )
        # set profiles
        prof_dict = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory=cassandra.query.dict_factory,
        )
        prof_tuple = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory=cassandra.query.tuple_factory,
        )
        profs = {"dict": prof_dict, "tuple": prof_tuple}
        # init cluster
        if cass_conf.cloud_config:
            self.cluster = Cluster(
                cloud=cass_conf.cloud_config,
                execution_profiles=profs,
                protocol_version=4,
                auth_provider=auth_prov,
            )
        else:
            if cass_conf.use_ssl:
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
                if cass_conf.ssl_certificate:
                    ssl_context.load_verify_locations(cass_conf.ssl_certificate)
                    ssl_context.verify_mode = ssl.CERT_REQUIRED
                if cass_conf.ssl_own_certificate and cass_conf.ssl_own_key:
                    ssl_context.load_cert_chain(
                        certfile=cass_conf.ssl_own_certificate,
                        keyfile=cass_conf.ssl_own_key,
                        password=cass_conf.ssl_own_key_pass,
                    )
            else:
                ssl_context = None
            self.cluster = Cluster(
                contact_points=cass_conf.cassandra_ips,
                execution_profiles=profs,
                protocol_version=4,
                auth_provider=auth_prov,
                port=cass_conf.cassandra_port,
                ssl_context=ssl_context,
            )
        self.cluster.connect_timeout = 10  # seconds
        # start session
        self.sess = self.cluster.connect()

    def __del__(self):
        self.cluster.shutdown()
