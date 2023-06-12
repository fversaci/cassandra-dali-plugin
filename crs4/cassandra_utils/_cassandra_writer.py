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
        table_data,
        table_metadata,
        id_col,
        label_col,
        data_col,
        cols,
        get_data,
        masks=False,
    ):
        self.get_data = get_data
        self.masks = masks
        self.table_data = table_data
        self.table_metadata = table_metadata
        self.id_col = id_col
        self.label_col = label_col
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
