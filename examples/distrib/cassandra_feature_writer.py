import io
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
            
        query1 = f"INSERT INTO {table_data} ("
        query1 += f"{id_col}, {label_col}, {data_col}) VALUES (?,?,?)"
        
        query2 = f"INSERT INTO {table_metadata} ("
        query2 += f"{id_col}, {label_col}) " 
        query2 += f"VALUES (?, ?)"

        self.prep1 = self.sess.prepare(query1)
        self.prep2 = self.sess.prepare(query2)

    
    def save_features(self, patch_id, label, features):
        partition_items = []
        features = self.get_data(features)
        item = (patch_id, label, features, partition_items)
        #print (f"Saving item: {patch_id, label}")
        self.save_item(item)


def get_cassandra_feature_writer(keyspace, table_suffix):
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
            id_col="patch_id",
            label_col="label",
            data_col="data",
            cols=[],
            get_data=get_data(),
        )

    return cw
