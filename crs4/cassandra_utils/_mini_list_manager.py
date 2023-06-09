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

from crs4.cassandra_utils._list_manager import ListManager

# pip3 install cassandra-driver
import cassandra
import ssl
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import TokenAwarePolicy, DCAwareRoundRobinPolicy
from cassandra.cluster import ExecutionProfile


class MiniListManager(ListManager):
    def __init__(
        self,
        cass_conf,
        use_ssl=False,
    ):
        """Loads the list of images from Cassandra DB

        :param cass_conf: Configuration for Cassandra
        :param use_ssl: Should use SSL connection?
        :returns:
        :rtype:

        """
        super().__init__()
        auth_prov = PlainTextAuthProvider(
            username=cass_conf.username, password=cass_conf.password)
        cassandra_ips=cass_conf.cassandra_ips
        cloud_config=cass_conf.cloud_config
        port=cass_conf.cassandra_port

        # cassandra parameters
        prof_dict = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory=cassandra.query.dict_factory,
        )
        prof_tuple = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory=cassandra.query.tuple_factory,
        )
        profs = {"dict": prof_dict, "tuple": prof_tuple}
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
        self.sess = self.cluster.connect()
        self.table = None
        self.id_col = None
        self._rows = None

    def __del__(self):
        self.cluster.shutdown()

    def set_config(self, conf):
        """Loads the list of images from Cassandra DB

        :param table: Matadata table with ids
        :param id_col: Cassandra id column for the images (e.g., 'image_id')
        :returns:
        :rtype:

        """
        super().__init__()
        self.table = conf["table"]
        self.id_col = conf["id_col"]

    def get_config(self):
        conf = {
            "table": self.table,
            "id_col": self.id_col,
        }
        return conf

    def read_rows_from_db(self):
        # get list of all rows
        query = f"SELECT {self.id_col} FROM {self.table} ;"
        res = self.sess.execute(query, execution_profile="tuple")
        all_ids = res.all()
        self.row_keys = list(map(lambda x: x[0], all_ids))

