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
import random


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
        bucket_col="bucket",
        buckets=None,
    ):
        self.buckets = buckets
        self.get_data = get_data
        prof = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory=cassandra.query.dict_factory,
        )
        prof_tuple = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory=cassandra.query.tuple_factory,
        )
        profs = {"default": prof, "tuple": prof_tuple}
        self.cluster = Cluster(
            cassandra_ips,
            execution_profiles=profs,
            protocol_version=4,
            auth_provider=auth_prov,
        )
        self.sess = self.cluster.connect()
        query1 = f"INSERT INTO {table_data} ("
        if self.buckets:
            query1 += f"{bucket_col}, "
        query1 += f"{id_col}, {label_col}, {data_col}) VALUES (?,?,?"
        if self.buckets:
            query1 += f",?"
        query1 += f")"
        query2 = f"INSERT INTO {table_metadata} ("
        if self.buckets:
            query2 += f"{bucket_col}, "
        query2 += f"{id_col}, {label_col}, {','.join(cols)}) "
        query2 += f"VALUES ({', '.join(['?']*(len(cols)+2))}"
        if self.buckets:
            query2 += f",?"
        query2 += f")"
        print(query1)
        print(query2)
        self.prep1 = self.sess.prepare(query1)
        self.prep2 = self.sess.prepare(query2)
        # query cardinality of buckets
        if self.buckets:
            query3 = f"SELECT COUNT(*) FROM {table_metadata} "
            query3 += f"WHERE {bucket_col} =?"
            self.prep3 = self.sess.prepare(query3)

    def __del__(self):
        self.cluster.shutdown()

    def save_item(self, item):
        if self.buckets:
            self.save_item_bucketing(item)
        else:
            self.save_item_normal(item)

    def save_item_normal(self, item):
        image_id, label, data, partition_items = item
        # insert metadata
        self.sess.execute(
            self.prep2,
            (image_id, label, *partition_items),
            execution_profile="default",
            timeout=30,
        )
        # insert heavy data
        self.sess.execute(
            self.prep1,
            (image_id, label, data),
            execution_profile="default",
            timeout=30,
        )

    def async_count(self, id):
        res = self.sess.execute_async(self.prep3, (id,),
                                      execution_profile="tuple", timeout=30,)
        return res
    
    def save_item_bucketing(self, item):
        image_id, label, data, partition_items = item
        # choose a random bucket, power-of-two choices
        choices = sorted([random.choice(self.buckets) for _ in range(2)])
        res = map(self.async_count, choices)
        res = list(map(lambda x: x.result().one()[0], res))
        if res[1]<res[0]:
            bucket_id = choices[1]
        else:
            bucket_id = choices[0]
        # insert metadata
        self.sess.execute(
            self.prep2,
            (bucket_id, image_id, label, *partition_items),
            execution_profile="default",
            timeout=30,
        )
        # insert heavy data
        self.sess.execute(
            self.prep1,
            (bucket_id, image_id, label, data),
            execution_profile="default",
            timeout=30,
        )

    def save_image(self, path, label, partition_items):
        # read file into memory
        data = self.get_data(path)
        image_id = uuid.uuid4()
        item = (image_id, label, data, partition_items)
        self.save_item(item)
