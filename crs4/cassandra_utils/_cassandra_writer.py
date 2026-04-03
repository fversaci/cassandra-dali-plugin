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

from crs4.cassandra_utils._cassandra_session import CassandraSession


class CassandraWriter:
    """Base class for writing data and metadata to Cassandra tables.

    This abstract base class handles the common setup for writing binary
    data (e.g., images) and associated metadata to separate Cassandra
    tables. Subclasses must implement set_query() and save_item() to
    define the specific schema and write logic.

    Attributes:
        get_data: Callable that reads file data into bytes.
        data_table: Name of the table storing binary data.
        metadata_table: Name of the table storing metadata.
        data_id_col: UUID column name in the data table.
        data_label_col: Label column name in the data table.
        metadata_id_col: UUID column name in the metadata table.
        metadata_label_col: Label column name in the metadata table.
        data_col: Binary data column name.
        cols: Additional metadata column names.
        sess: Active Cassandra session.
    """
    def __init__(
        self,
        cass_conf,
        data_table,
        metadata_table,
        data_id_col,
        data_label_col,
        data_col,
        cols,
        get_data,
        metadata_id_col=None,
        metadata_label_col=None,
    ):
        """Initialize CassandraWriter with table and column configuration.

        Args:
            cass_conf: CassandraConf object with connection parameters.
            data_table: Name of the data table (keyspace.tablename).
            metadata_table: Name of the metadata table (keyspace.tablename).
            data_id_col: Column name for UUIDs in the data table.
            data_label_col: Column name for labels in the data table.
            data_col: Column name for binary data (BLOB).
            cols: List of additional metadata column names.
            get_data: Callable(path) -> bytes that reads file content.
            metadata_id_col: Column name for UUIDs in metadata table.
                Defaults to data_id_col if None.
            metadata_label_col: Column name for labels in metadata table.
                Defaults to data_label_col if None.

        Raises:
            ValueError: If any required parameter is missing or empty.
        """
        # Validate required parameters
        if cass_conf is None:
            raise ValueError("cass_conf cannot be None")
        if not data_table:
            raise ValueError("data_table cannot be empty")
        if not metadata_table:
            raise ValueError("metadata_table cannot be empty")
        if not data_id_col:
            raise ValueError("data_id_col cannot be empty")
        if not data_label_col:
            raise ValueError("data_label_col cannot be empty")
        if not data_col:
            raise ValueError("data_col cannot be empty")
        if cols is None:
            raise ValueError("cols cannot be None")
        if get_data is None:
            raise ValueError("get_data cannot be None")

        self.get_data = get_data
        self.data_table = data_table
        self.metadata_table = metadata_table
        self.data_id_col = data_id_col
        self.data_label_col = data_label_col

        if metadata_id_col:
            self.metadata_id_col = metadata_id_col
        else:
            self.metadata_id_col = data_id_col

        if metadata_label_col:
            self.metadata_label_col = metadata_label_col
        else:
            self.metadata_label_col = data_label_col

        self.data_col = data_col
        self.cols = cols
        self._cs = CassandraSession(cass_conf)
        self.sess = self._cs.sess

        # Query and session prepare have to be implemented
        # in subclasses set_query() method as well as
        # session execute in subclass save_item method
        self.set_query()

    def set_query(self):
        """Prepare INSERT statements for data and metadata tables.

        Subclasses must implement this method to prepare the appropriate
        CQL statements (self.prep_data, self.prep_meta) based on their
        specific schema.
        """
        pass

    def save_item(self, item):
        """Insert a single item (metadata and binary data) into Cassandra.

        Subclasses must implement this method to execute the prepared
        statements with data extracted from the item tuple.

        Args:
            item: Tuple containing (id, label, data, partition_items).
        """
        pass
