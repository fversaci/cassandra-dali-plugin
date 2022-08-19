# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from crs4.cassandra_utils._list_manager import ListManager

# pip3 install cassandra-driver
import cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import TokenAwarePolicy, DCAwareRoundRobinPolicy
from cassandra.cluster import ExecutionProfile


class MiniListManager(ListManager):
    def __init__(
        self,
        auth_prov,
        cassandra_ips,
        port=9042,
    ):
        """Loads the list of images from Cassandra DB

        :param auth_prov: Authenticator for Cassandra
        :param cassandra_ips: List of Cassandra ip's
        :param port: Cassandra server port (default: 9042)
        :returns:
        :rtype:

        """
        super().__init__()
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
        self.cluster = Cluster(
            cassandra_ips,
            execution_profiles=profs,
            protocol_version=4,
            auth_provider=auth_prov,
            port=port,
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

