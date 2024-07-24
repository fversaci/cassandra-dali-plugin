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
        # set query and prepare
        pass

    def save_item(self, item):
        # insert metadata and heavy data
        pass
