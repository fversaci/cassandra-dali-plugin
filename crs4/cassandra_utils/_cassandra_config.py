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


class CassandraConf:
    def __init__(self):
        self.username = None
        self.password = None
        self.cloud_config = None
        self.cassandra_ips = None
        self.cassandra_port = 9042
        self.use_ssl = False
        self.ssl_certificate = ""  # "server.crt"
        self.ssl_own_certificate = ""  # "client.crt"
        self.ssl_own_key = ""  # "client.key"
        self.ssl_own_key_pass = ""  # "key-password"
