# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import cassandra
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from cassandra.cluster import ExecutionProfile
from cassandra.policies import TokenAwarePolicy, DCAwareRoundRobinPolicy
import uuid


class CassandraWriter:
    def __init__(
        self,
        auth_prov,
        cassandra_ips,
        table_ids,
        table_data,
        table_metadata,
        id_col,
        label_col,
        data_col,
        cols,
        get_data,
    ):
        self.get_data = get_data
        prof = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory=cassandra.query.dict_factory,
        )
        profs = {"default": prof}
        self.cluster = Cluster(
            cassandra_ips,
            execution_profiles=profs,
            protocol_version=4,
            auth_provider=auth_prov,
        )
        self.sess = self.cluster.connect()
        query1 = (
            f"INSERT INTO {table_ids} "
            + f"({id_col}, {label_col}, {', '.join(cols)}) "
            + f"VALUES ({', '.join(['?']*(len(cols)+2))})"
        )
        query2 = (
            f"INSERT INTO {table_data} "
            + f"({id_col}, {label_col}, {data_col}) VALUES (?,?,?)"
        )
        query3 = (
            f"INSERT INTO {table_metadata} "
            + f"({id_col}, {label_col}, {', '.join(cols)}) "
            + f"VALUES ({', '.join(['?']*(len(cols)+2))})"
        )
        self.prep1 = self.sess.prepare(query1)
        self.prep2 = self.sess.prepare(query2)
        self.prep3 = self.sess.prepare(query3)

    def __del__(self):
        self.cluster.shutdown()

    def save_item(self, item):
        image_id, label, data, partition_items = item
        i1 = self.sess.execute_async(
            self.prep1,
            (image_id, label, *partition_items),
            execution_profile="default",
            timeout=30,
        )
        i3 = self.sess.execute_async(
            self.prep3,
            (image_id, label, *partition_items),
            execution_profile="default",
            timeout=30,
        )
        # wait for previous inserts to finish
        i1.result()
        i3.result()
        # insert heavy data synchronously
        self.sess.execute(
            self.prep2, (image_id, label, data), execution_profile="default", timeout=30
        )

    def save_image(self, path, label, partition_items):
        # read file into memory
        data = self.get_data(path)
        image_id = uuid.uuid4()
        item = (image_id, label, data, partition_items)
        self.save_item(item)
