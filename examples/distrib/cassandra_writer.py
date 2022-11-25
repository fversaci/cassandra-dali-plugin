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
        masks=False,
    ):
        print ("DD: %r" % get_data) 
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
            print (type(cloud_config))
            self.cluster = Cluster(
                cloud=cloud_config,
                execution_profiles=profs,
                protocol_version=4,
                auth_provider=auth_prov,
            )
        else:
            self.cluster = Cluster(
                contact_points=cassandra_ips,
                port=cassandra_port,
                execution_profiles=profs,
                protocol_version=4,
                auth_provider=auth_prov,
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
        # insert metadata 
        pass
        # insert heavy data 
        pass 
