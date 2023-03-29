import cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import TokenAwarePolicy, DCAwareRoundRobinPolicy
from cassandra.cluster import ExecutionProfile
from cassandra import ConsistencyLevel

import pandas as pd
import numpy as np
import pickle 

def pandas_factory(colnames, rows):
    return pd.DataFrame(rows, columns=colnames)
    

### Base class 

class split_generator:
    def __init__(self, id_col=None, data_col=None, label_col=None, label_type=None):
        ## Preliminary check on arguments
        if label_type != 'none' and not label_col:
            raise Exception("Please provide the label_col argument")
        self._df = None
        self._id_col = id_col
        self._data_col = data_col
        self._label_col = label_col
        self._label_type = label_type
        self._data_table = None
        self._metadata_table = None
        
    def load_from_db(self, CC, keyspace, table_suffix):
        self._keyspace = keyspace
        self._table_suffix = table_suffix
        self._data_table = f'{keyspace}.data_{table_suffix}'
        self._metadata_table = f'{keyspace}.metadata_{table_suffix}'
        self.CC = CC
        self._df = self.get_df_from_metadata()
        self.setup()
        
    def load_from_file(self, fn):
        dict_tmp = pickle.load(open(fn, 'rb'))
        self._keyspace = dict_tmp['keyspace']
        self._table_suffix =dict_tmp['table_suffix']
        self._data_table = dict_tmp['data_table']
        self._metadata_table = dict_tmp['metadata_table']
        self._df = dict_tmp['df']
        self.setup()
        
    def cache_db_data_to_file(self, fn):
        if not isinstance(self._df, pd.DataFrame) or not self._data_table or not self._metadata_table:
            raise Exception('No dataframe defined yet.')
        dict_tmp = {'keyspace': self._keyspace,
                    'table_suffix': self._table_suffix,
                    'data_table': self._data_table, 
                    'metadata_table': self._metadata_table,
                    'df': self._df}
        
        pickle.dump(dict_tmp, open(fn, 'wb'))
            
    def setup(self):
        self.split_metadata = {'keyspace': self._keyspace,
                               'table_suffix': self._table_suffix,
                               'id_col': self._id_col,
                               'data_col': self._data_col,
                               'label_type': self._label_type,  # String {int|image|none} to be defined in derived classes
                               'label_col': self._label_col,
                               'row_keys': np.empty(1),# 1D Numpy array containing UUIDs. 
                                                       # Computed in derived classes. 
                                                       # This is just a placeholder initialization.
                               'split': [np.empty(1), np.empty(1)]# List of 1D Numpy arrays. Each array represent a single split. 
                                                                  #Each element of the array is the index of a correspondig UUID in row_keys
                                                                  # split structure must be computed in derived classes. 
                                                                  #This is just a placeholder initialization
                              } 

    def get_df_from_metadata(self):
        # Load list of uuids and cols from Cassandra DB...
        ap = PlainTextAuthProvider(username=self.CC.username, password=self.CC.password)

        # pandas profile
        prof_pandas = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory=pandas_factory,
        )

        profs = {"pandas": prof_pandas}

        cluster = Cluster(
                        contact_points=self.CC.cassandra_ips,
                        execution_profiles=profs,
                        protocol_version=4,
                        auth_provider=ap,
                        port=self.CC.cassandra_port,
                    )
        cluster.connect_timeout = 10  # seconds
        sess=cluster.connect()
        sess.default_fetch_size = 10000000

        ## Get rows
        query = f"SELECT * FROM {self._metadata_table};"

        res = sess.execute(query, execution_profile="pandas", timeout=60)
        df = res._current_rows

        sess.shutdown()

        return df 
    
    def save_splits(self, out_split_fn='cassandra_split_file.pckl'):
        pickle.dump(self.split_metadata, open(out_split_fn, "wb"))
        
    def create_splits(self, **kwargs):
        """
        This must be implemented in derived classes
        """
        None

