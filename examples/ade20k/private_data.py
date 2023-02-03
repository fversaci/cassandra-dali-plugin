# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from cassandra_config import CassandraConf

CassConf = CassandraConf()
CassConf.username = "guest"
CassConf.password = "test"
CassConf.cassandra_ips = ["127.0.0.1"]
# CassConf.cassandra_port = 9042
# CassConf.cloud_config = {'secure_connect_bundle': 'tmp/secure-connect-ade20k.zip'}
