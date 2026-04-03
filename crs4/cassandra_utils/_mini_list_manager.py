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
    """Manages UUID lists by loading them from Cassandra metadata tables.

    This class connects to Cassandra, queries a metadata table for UUIDs,
    and provides methods to save/load the UUID list to/from local files.
    It stores configuration (table name, column name) so the same query
    can be reproduced when loading from cache.

    Attributes:
        table: Name of the Cassandra metadata table.
        id_col: Name of the column containing UUIDs.
        sess: Cassandra session object.
    """
    def __init__(
        self,
        cass_conf,
    ):
        """Initialize MiniListManager with Cassandra configuration.

        Args:
            cass_conf: CassandraConf object with connection parameters
                (IPs, port, credentials, SSL settings, etc.).
        """
        super().__init__()
        self._cs = CassandraSession(cass_conf)
        self.sess = self._cs.sess
        self.table = None
        self.id_col = None
        self._rows = None

    def set_config(self, conf):
        """Set configuration from a dictionary.

        Restores the table name and ID column from a configuration
        dictionary (typically loaded from a cached file).

        Args:
            conf: Dictionary with keys 'table' (metadata table name) and
                'id_col' (UUID column name).
        """
        self.row_keys = None
        self.split = None
        self.table = conf["table"]
        self.id_col = conf["id_col"]

    def get_config(self):
        """Get current configuration as a dictionary.

        Returns:
            dict: Dictionary with 'table' and 'id_col' keys.
        """
        conf = {
            "table": self.table,
            "id_col": self.id_col,
        }
        return conf

    def read_rows_from_db(self):
        """Load UUIDs from the configured Cassandra metadata table.

        Executes a SELECT query to retrieve all UUID values from the
        configured table and column, storing them in self.row_keys.
        """
        # get list of all rows
        query = f"SELECT {self.id_col} FROM {self.table} ;"
        res = self.sess.execute(query, execution_profile="tuple")
        all_ids = res.all()
        self.row_keys = list(map(lambda x: x[0], all_ids))
