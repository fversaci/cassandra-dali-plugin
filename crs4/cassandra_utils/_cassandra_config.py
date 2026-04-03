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
    """Configuration container for Cassandra database connections.

    This class holds all connection parameters needed to connect to a
    Cassandra cluster, including authentication, SSL/TLS settings, and
    cloud configuration (e.g., for Astra DB).

    Attributes:
        username: Cassandra username for authentication.
        password: Cassandra password for authentication.
        cloud_config: Dictionary for cloud-based connections (e.g.,
            {'secure_connect_bundle': 'path/to/bundle.zip'} for Astra).
        cassandra_ips: List of Cassandra node IP addresses or hostnames.
        cassandra_port: Cassandra native transport port (default: 9042).
        use_ssl: Whether to use SSL/TLS for the connection.
        ssl_certificate: Path to the server's CA certificate file.
        ssl_own_certificate: Path to the client certificate file.
        ssl_own_key: Path to the client private key file.
        ssl_own_key_pass: Password for the client private key.
    """
    def __init__(self):
        """Initialize with default values."""
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
