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

from cassandra_config import CassandraConf

CassConf = CassandraConf()
CassConf.username = "guest"
CassConf.password = "test"
CassConf.cassandra_ips = ["127.0.0.1"]
# CassConf.cassandra_port = 9042
# CassConf.cloud_config = {'secure_connect_bundle': 'secure-connect-blabla.zip'}
