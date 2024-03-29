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
from crs4.cassandra_utils._cassandra_session import CassandraSession


class MiniListManager(ListManager):
    def __init__(
        self,
        cass_conf,
    ):
        """Loads the list of images from Cassandra DB

        :param cass_conf: Configuration for Cassandra
        :returns:
        :rtype:
        """
        super().__init__()
        self._cs = CassandraSession(cass_conf)
        self.sess = self._cs.sess
        self.table = None
        self.id_col = None
        self._rows = None

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
