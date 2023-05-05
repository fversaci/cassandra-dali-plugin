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
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from cassandra.cluster import ExecutionProfile
from cassandra.policies import TokenAwarePolicy, DCAwareRoundRobinPolicy
import uuid

from crs4.cassandra_utils._cassandra_writer import CassandraWriter


class CassandraSegmentationWriter(CassandraWriter):
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

        super().__init__(
            auth_prov,
            table_data,
            table_metadata,
            id_col,
            label_col,
            data_col,
            cols,
            get_data,
            cloud_config,
            cassandra_ips,
            cassandra_port,
            use_ssl,
            masks,
        )

    def set_query(self):
        query_data = f"INSERT INTO {self.table_data} ("
        query_data += (
            f"{self.id_col}, {self.label_col}, {self.data_col}) VALUES (?,?,?)"
        )
        query_meta = f"INSERT INTO {self.table_metadata} ("
        query_meta += f"{self.id_col}, {', '.join(self.cols)}) "
        query_meta += f"VALUES ({', '.join(['?']*(len(self.cols)+1))})"

        self.prep_data = self.sess.prepare(query_data)
        self.prep_meta = self.sess.prepare(query_meta)

    def __del__(self):
        self.cluster.shutdown()

    def save_item(self, item):
        image_id, label, data, partition_items = item
        stuff = (image_id, *partition_items)
        # insert metadata
        self.sess.execute(
            self.prep_meta,
            stuff,
            execution_profile="default",
            timeout=30,
        )
        # insert heavy data
        self.sess.execute(
            self.prep_data,
            (image_id, label, data),
            execution_profile="default",
            timeout=30,
        )

    def save_image(self, path, label, partition_items):
        # read file into memory
        data = self.get_data(path)
        label = self.get_data(label)
        image_id = uuid.uuid4()
        item = (image_id, label, data, partition_items)
        self.save_item(item)
