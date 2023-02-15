# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from crs4.cassandra_utils._mini_list_manager import MiniListManager

class ISUP_ListManager(MiniListManager):
    def __init__(
        self,
        auth_prov,
        cassandra_ips= None,
        cloud_config=None,
        port=None,
    ):
        super().__init__(
                auth_prov,
                cassandra_ips,
                cloud_config,
                port)
        
        self.row_keys_labs = None
    
    def read_rows_from_db_id_labs(self, cols=['label']):
        # get list of all rows and their labels
        if cols:
            col_str = ','.join(cols)
            query = f"SELECT {self.id_col},{col_str} FROM {self.table} ;"
            res = self.sess.execute(query, execution_profile="tuple")
            all_ = res.all()
            self.row_keys_labs = list(all_)
        else:
            self.read_rows_from_db()
