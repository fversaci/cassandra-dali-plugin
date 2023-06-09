# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from cassandra_config import CassandraConf

cass_conf = CassandraConf()
cass_conf.username = "guest"
cass_conf.password = "test"
cass_conf.cassandra_ips = ["127.0.0.1"]
# cass_conf.cassandra_port = 9042
# cass_conf.cloud_config = {'secure_connect_bundle': 'secure-connect-blabla.zip'}
