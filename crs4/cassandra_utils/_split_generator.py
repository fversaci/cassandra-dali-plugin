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
import pandas as pd
import numpy as np
import pickle


class split_generator:
    """Base class for generating train/validation/test splits from Cassandra metadata.

    This class loads metadata from Cassandra, applies filtering and splitting
    logic (to be implemented in subclasses), and caches the results for later
    use by the DALI data loader.

    Attributes:
        _df: Pandas DataFrame containing the metadata.
        _metadata_id_col: Column name for IDs in the metadata table.
        _data_id_col: Column name for IDs in the data table.
        _metadata_label_col: Column name for labels in the metadata table.
        _data_label_col: Column name for labels in the data table.
        _label_type: Type of labels ("int", "blob", or "none").
        _data_col: Column name for binary data.
        _data_table: Name of the data table.
        _metadata_table: Name of the metadata table.
        split_metadata: Dictionary containing split configuration and results.
    """
    def __init__(
        self,
        data_id_col=None,
        metadata_id_col=None,
        data_col=None,
        data_label_col=None,
        metadata_label_col=None,
        label_type=None,
    ):
        """Initialize split generator with column mappings.

        Args:
            data_id_col: Column name for IDs in the data table. If None,
                defaults to metadata_id_col.
            metadata_id_col: Column name for IDs in the metadata table.
            data_col: Column name for binary data in the data table.
            data_label_col: Column name for labels in the data table.
            metadata_label_col: Column name for labels in the metadata table.
            label_type: Type of labels - "int" for classification, "blob" for
                segmentation masks, or "none" for no labels.

        Raises:
            Exception: If label_type is not "none" and metadata_label_col is
                not provided.
        """
        ## Preliminary check on arguments
        if label_type != "none" and not metadata_label_col:
            raise Exception("Please provide the label_col argument")

        self._df = None
        self._metadata_id_col = metadata_id_col
        if data_id_col:
            self._data_id_col = data_id_col
        else:
            self._data_id_col = metadata_id_col

        self._metadata_label_col = metadata_label_col
        if data_label_col:
            self._data_label_col = data_label_col
        else:
            self._data_label_col = metadata_label_col

        self._label_type = label_type

        self._data_col = data_col
        self._data_table = None
        self._metadata_table = None

    def load_from_db(self, cass_conf, data_table, metadata_table):
        """Load metadata from Cassandra and initialize the split.

        Args:
            cass_conf: CassandraConf object with connection parameters.
            data_table: Name of the data table (keyspace.tablename).
            metadata_table: Name of the metadata table (keyspace.tablename).
        """
        self._data_table = data_table
        self._metadata_table = metadata_table
        self.cass_conf = cass_conf
        self._df = self.get_df_from_metadata()
        self.setup()

    def load_from_file(self, fn):
        """Load previously cached metadata from a pickle file.

        Args:
            fn: Path to the pickle file created by cache_db_data_to_file.
        """
        dict_tmp = pickle.load(open(fn, "rb"))
        self._data_table = dict_tmp["data_table"]
        self._metadata_table = dict_tmp["metadata_table"]
        self._df = dict_tmp["df"]
        self.setup()

    def cache_db_data_to_file(self, fn):
        """Save the metadata DataFrame to a pickle file for later reuse.

        Args:
            fn: Path to the output pickle file.

        Raises:
            Exception: If no DataFrame has been loaded yet (call load_from_db first).
        """
        if (
            not isinstance(self._df, pd.DataFrame)
            or not self._data_table
            or not self._metadata_table
        ):
            raise Exception("No dataframe defined yet.")
        dict_tmp = {
            "data_table": self._data_table,
            "metadata_table": self._metadata_table,
            "df": self._df,
        }

        pickle.dump(dict_tmp, open(fn, "wb"))

    def setup(self):
        """Initialize the split_metadata dictionary with column mappings and placeholders.

        Populates split_metadata with table/column names, label type, and
        placeholder arrays for row_keys and splits. Subclasses should
        override create_splits() to compute the actual split indices.
        """
        self.split_metadata = {
            "data_table": self._data_table,
            "data_id_col": self._data_id_col,
            "data_label_col": self._data_label_col,
            "metadata_table": self._metadata_table,
            "metadata_id_col": self._metadata_id_col,
            "metadata_label_col": self._metadata_label_col,
            "data_col": self._data_col,
            "label_type": self._label_type,  # String {int|blob|none} to be defined in derived classes
            "row_keys": np.empty(1),  # 1D Numpy array containing UUIDs.
            # Computed in derived classes.
            # This is just a placeholder initialization.
            "split": [
                np.empty(1),
                np.empty(1),
            ],  # List of 1D Numpy arrays. Each array represent a single split.
            # Each element of the array is the index of a correspondig UUID in row_keys
            # split structure must be computed in derived classes.
            # This is just a placeholder initialization
        }

    def get_df_from_metadata(self):
        """Fetch all rows from the metadata table and return as a DataFrame.

        Returns:
            pandas.DataFrame: All rows from the configured metadata table.
        """
        cs = CassandraSession(self.cass_conf)
        sess = cs.sess

        ## Get rows
        query = f"SELECT * FROM {self._metadata_table};"
        rows = sess.execute(query, execution_profile="dict", timeout=120)
        df = pd.DataFrame(rows)

        return df

    def save_splits(self, out_split_fn="cassandra_split_file.pckl"):
        """Save the split_metadata dictionary to a pickle file.

        Args:
            out_split_fn: Output filename (default: "cassandra_split_file.pckl").
        """
        pickle.dump(self.split_metadata, open(out_split_fn, "wb"))

    def create_splits(self, **kwargs):
        """
        This must be implemented in derived classes
        """
        raise NotImplementedError("Subclasses must implement create_splits")
