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

# pip3 install cassandra-driver
import cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import TokenAwarePolicy, DCAwareRoundRobinPolicy
from cassandra.cluster import ExecutionProfile
import ssl


class CassandraSession:
    """Manages a session connection to a Cassandra database.

    This class wraps the Cassandra Python driver's Cluster and Session
    objects, handling authentication, SSL/TLS configuration, and cloud
    connection bundles (e.g., for Astra DB).

    Attributes:
        cluster: The Cassandra Cluster object.
        sess: The active Cassandra Session object.

    Example:
        >>> conf = CassandraConf()
        >>> conf.cassandra_ips = ["localhost"]
        >>> session = CassandraSession(conf)
        >>> results = session.sess.execute("SELECT * FROM keyspace.table")
    """
    def __init__(self, cass_conf):
        """Create a new Cassandra session.

        Args:
            cass_conf: CassandraConf object containing connection parameters.

        Raises:
            ValueError: If cass_conf is None or missing required attributes.
        """
        # Validate configuration object
        if cass_conf is None:
            raise ValueError("cass_conf cannot be None")

        required_attrs = [
            "username",
            "password",
            "cloud_config",
            "use_ssl",
            "ssl_certificate",
            "ssl_own_certificate",
            "ssl_own_key",
            "ssl_own_key_pass",
            "cassandra_ips",
            "cassandra_port",
        ]

        for attr in required_attrs:
            if not hasattr(cass_conf, attr):
                raise ValueError(f"Missing required configuration attribute: {attr}")

        # read parameters
        auth_prov = PlainTextAuthProvider(
            username=cass_conf.username, password=cass_conf.password
        )
        # set profiles
        prof_dict = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory=cassandra.query.dict_factory,
        )
        prof_tuple = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory=cassandra.query.tuple_factory,
        )
        profs = {"dict": prof_dict, "tuple": prof_tuple}
        # init cluster
        if cass_conf.cloud_config:
            self.cluster = Cluster(
                cloud=cass_conf.cloud_config,
                execution_profiles=profs,
                protocol_version=4,
                auth_provider=auth_prov,
            )
        else:
            if cass_conf.use_ssl:
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
                if cass_conf.ssl_certificate:
                    ssl_context.load_verify_locations(cass_conf.ssl_certificate)
                    ssl_context.verify_mode = ssl.CERT_REQUIRED
                if cass_conf.ssl_own_certificate and cass_conf.ssl_own_key:
                    ssl_context.load_cert_chain(
                        certfile=cass_conf.ssl_own_certificate,
                        keyfile=cass_conf.ssl_own_key,
                        password=cass_conf.ssl_own_key_pass,
                    )
            else:
                ssl_context = None
            self.cluster = Cluster(
                contact_points=cass_conf.cassandra_ips,
                execution_profiles=profs,
                protocol_version=4,
                auth_provider=auth_prov,
                port=cass_conf.cassandra_port,
                ssl_context=ssl_context,
            )
        self.cluster.connect_timeout = 10  # seconds
        # start session
        self.sess = self.cluster.connect()

    def __del__(self):
        """Clean up resources by shutting down the cluster connection."""
        self.cluster.shutdown()
