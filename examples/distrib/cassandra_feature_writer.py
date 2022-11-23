import io,sys
import cassandra
from cassandra.auth import PlainTextAuthProvider
from cassandra_writer import CassandraWriter

import getpass
from tqdm import tqdm
import io
import numpy as np

def get_data():
    def r(feature_tensor):
        data = io.BytesIO(feature_tensor.cpu().numpy())
        return data.getvalue()
    return r


class CassandraFeatureWriter(CassandraWriter):
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
        self.table_data = table_data
        self.table_metadata = table_metadata
        self.id_col = id_col
        self.label_col = label_col
        self.data_col = data_col

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
                masks
                )

    def set_query(self):
        query_data = f"INSERT INTO {self.table_data} ("
        query_data += f"{self.id_col},{self.label_col},{self.data_col},sample_name,sample_rep) VALUES (?,?,?,?,?)"
        
        query_meta = f"INSERT INTO {self.table_metadata} ("
        query_meta += f"{self.id_col},{self.label_col},sample_name,sample_rep) " 
        query_meta += f"VALUES (?,?,?,?)"

        self.prep_data = self.sess.prepare(query_data)
        self.prep_meta = self.sess.prepare(query_meta)

    def save_item(self, item):
        image_id, label, data, partition_items = item
        stuff = (image_id,label,*partition_items)
        # insert metadata 
        self.sess.execute(
            self.prep_meta,
            stuff,
            execution_profile="default",
            timeout=30,
        )
        # insert heavy data 
        self.sess.execute(
            self.prep_data, (image_id, label, data, *partition_items),
            execution_profile="default", timeout=30,
        )
        
    def save_features(self, patch_id, label, features, partition_items=[]):
        features = self.get_data(features)
        item = (patch_id, label, features, partition_items)
        self.save_item(item)


def get_cassandra_feature_writer(keyspace, table_suffix, id_col='patch_id', data_col='data', label_col='label', cols=[]):
    # Read Cassandra parameters
    try:
        from private_data import CassConf as CC
        cloud_config=CC.cloud_config
        cassandra_ips=CC.cassandra_ips
        cassandra_port=CC.cassandra_port
        username=CC.username
        password=CC.password
    except ImportError:
        cassandra_ip = getpass("Insert Cassandra's IP address: ")
        cassandra_ips = [cassandra_ip]
        username = getpass("Insert Cassandra user: ")
        password = getpass("Insert Cassandra password: ")

    auth_prov = PlainTextAuthProvider(username, password)
    cw = CassandraFeatureWriter(
            cloud_config=cloud_config,
            auth_prov=auth_prov,
            cassandra_ips=cassandra_ips,
            cassandra_port=cassandra_port,
            table_data=f"{keyspace}.data_{table_suffix}",
            table_metadata=f"{keyspace}.metadata_{table_suffix}",
            id_col=id_col,
            label_col=label_col,
            data_col=data_col,
            cols=cols,
            get_data=get_data(),
        )

    return cw
