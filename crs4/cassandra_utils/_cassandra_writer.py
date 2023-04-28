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

import cassandra
import ssl
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from cassandra.cluster import ExecutionProfile
from cassandra.policies import TokenAwarePolicy, DCAwareRoundRobinPolicy


class CassandraWriter:
    def __init__(
        self,
        auth_prov,
        table_data,
        table_metadata,
        id_col,
        label_col,
        data_col,
        cols,
        get_data,
        cloud_config=None,
        cassandra_ips=None,
        cassandra_port=None,
        use_ssl=False,
        masks=False,
    ):
        self.get_data = get_data
        self.masks = masks
        self.table_data = table_data
        self.table_metadata = table_metadata
        self.id_col = id_col
        self.label_col = label_col
        self.data_col = data_col
        self.cols = cols

        prof = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory=cassandra.query.dict_factory,
        )
        profs = {"default": prof}

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
                port=cassandra_port,
                execution_profiles=profs,
                protocol_version=4,
                auth_provider=auth_prov,
                ssl_context=ssl_context,
            )
        self.sess = self.cluster.connect()

        # Query and session prepare have to be implemented
        # in subclasses set_query() method as well as
        # session execute in subclass save_item method
        self.set_query()

    def __del__(self):
        self.cluster.shutdown()

    def set_query(self):
        # set query and prepare
        pass

    def save_item(self, item):
        # insert metadata and heavy data
        pass
