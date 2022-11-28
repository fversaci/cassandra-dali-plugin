# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


class CassandraConf:
    def __init__(self):
        self.username = None
        self.password = None
        self.cloud_config = None
        self.cassandra_ips = None
        self.cassandra_port = 9042
