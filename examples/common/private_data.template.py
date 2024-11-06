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

from crs4.cassandra_utils import CassandraConf

cass_conf = CassandraConf()
cass_conf.username = "guest"
cass_conf.password = "test"
cass_conf.cassandra_ips = ["127.0.0.1"]
cass_conf.use_ssl = True
cass_conf.cassandra_port = 9042
# cass_conf.ssl_certificate = "/certs/rootca.crt"
# cass_conf.ssl_own_certificate = "/certs/client.crt"
# cass_conf.ssl_own_key = "/certs/client.key"
# cass_conf.cloud_config = {'secure_connect_bundle': 'secure-connect-blabla.zip'}
