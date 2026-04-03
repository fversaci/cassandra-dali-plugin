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
from cassandra import concurrent
from cassandra.query import BatchStatement
import uuid

from crs4.cassandra_utils._cassandra_writer import CassandraWriter
from crs4.cassandra_utils._cassandra_session import CassandraSession


class CassandraClassificationWriter(CassandraWriter):
    """Writer for image classification datasets.

    Stores images as BLOBs in a data table with integer labels, and
    metadata (e.g., original label name, split) in a separate metadata
    table. Supports both direct insertion and batched enqueueing for
    improved throughput.

    Attributes:
        queue_data: List of pending data table inserts.
        queue_meta: List of pending metadata table inserts.
        concurrency: Number of concurrent insert operations (default: 32).
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
        """Initialize classification writer.

        Args:
            cass_conf: CassandraConf object with connection parameters.
            data_table: Name of the data table (keyspace.tablename).
            metadata_table: Name of the metadata table (keyspace.tablename).
            data_id_col: UUID column name in the data table.
            data_label_col: Integer label column name in the data table.
            data_col: BLOB column name for image data.
            cols: List of additional metadata column names (e.g., ['or_split', 'or_label']).
            get_data: Callable(path) -> bytes that reads image file content.
            metadata_id_col: UUID column name in metadata table (defaults to data_id_col).
            metadata_label_col: Integer label column name in metadata table (defaults to data_label_col).
        """
        super().__init__(
            cass_conf=cass_conf,
            data_table=data_table,
            metadata_table=metadata_table,
            data_id_col=data_id_col,
            data_label_col=data_label_col,
            metadata_id_col=metadata_id_col,
            metadata_label_col=metadata_label_col,
            data_col=data_col,
            cols=cols,
            get_data=get_data,
        )
        self.queue_data = []
        self.queue_meta = []
        self.concurrency = 32

    def set_query(self):
        """Prepare INSERT statements for classification data and metadata."""
        query_data = f"INSERT INTO {self.data_table} ("
        query_data += f"{self.data_id_col}, {self.data_label_col}, {self.data_col}) VALUES (?,?,?)"
        query_meta = f"INSERT INTO {self.metadata_table} ("
        query_meta += f"{self.metadata_id_col}, {self.metadata_label_col}, {', '.join(self.cols)}) "
        query_meta += f"VALUES ({', '.join(['?'] * (len(self.cols) + 2))})"

        self.prep_data = self.sess.prepare(query_data)
        self.prep_meta = self.sess.prepare(query_meta)

    def save_item(self, item):
        """Insert a single image and its metadata synchronously.

        Args:
            item: Tuple of (image_id, label, data, partition_items) where
                image_id is a UUID, label is an integer, data is bytes,
                and partition_items are additional metadata values.
        """
        image_id, label, data, partition_items = item
        stuff = (image_id, label, *partition_items)

        batch = BatchStatement()
        # insert metadata
        batch.add(self.prep_meta, stuff)
        # insert heavy data
        batch.add(self.prep_data, (image_id, label, data))

        self.sess.execute(batch, execution_profile="tuple", timeout=30)

    def enqueue_item(self, item):
        """Queue an item for batched insertion.

        Items are accumulated in memory and sent when the queue reaches
        the concurrency limit (via send_enqueued).

        Args:
            item: Tuple of (image_id, label, data, partition_items).
        """
        image_id, label, data, partition_items = item
        stuff_meta = (image_id, label, *partition_items)
        stuff_data = (image_id, label, data)
        self.queue_meta += (stuff_meta,)
        self.queue_data += (stuff_data,)

    def send_enqueued(self):
        """Flush all queued inserts to the database.

        Data and metadata queues are sent separately using concurrent
        execution for improved throughput.
        """
        if self.queue_data:
            cassandra.concurrent.execute_concurrent_with_args(
                self.sess, self.prep_data, self.queue_data
            )
            self.queue_data = []
        if self.queue_meta:
            concurrent.execute_concurrent_with_args(
                self.sess, self.prep_meta, self.queue_meta, concurrency=self.concurrency
            )
            self.queue_meta = []

    def save_image(self, path, label, partition_items):
        """Read an image file and insert it directly to Cassandra.

        Args:
            path: Filesystem path to the image file.
            label: Integer class label.
            partition_items: Tuple of additional metadata values.
        """
        # read file into memory
        data = self.get_data(path)
        image_id = uuid.uuid4()
        item = (image_id, label, data, partition_items)
        self.save_item(item)

    def enqueue_image(self, path, label, partition_items):
        """Read an image file and queue it for batched insertion.

        Args:
            path: Filesystem path to the image file.
            label: Integer class label.
            partition_items: Tuple of additional metadata values.
        """
        # read file into memory
        data = self.get_data(path)
        image_id = uuid.uuid4()
        item = (image_id, label, data, partition_items)
        self.enqueue_item(item)
        if len(self.queue_meta) % self.concurrency == 0:
            self.send_enqueued()
