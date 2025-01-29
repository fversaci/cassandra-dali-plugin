#! /bin/bash -x
set -e

ssh root@cassandra 'SSL_VALIDATE=false /opt/cassandra/bin/cqlsh --ssl -e "DROP KEYSPACE IF EXISTS imagenette;"'
cat create_tables.cql | ssh root@cassandra 'SSL_VALIDATE=false /opt/cassandra/bin/cqlsh --ssl'
python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=train --data-table=imagenette.data_train --metadata-table=imagenette.metadata_train
python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=val --data-table=imagenette.data_val --metadata-table=imagenette.metadata_val
python3 cache_uuids.py --metadata-table=imagenette.metadata_train --rows-fn=train.rows
python3 cache_uuids.py --metadata-table=imagenette.metadata_val --rows-fn=val.rows
echo "--- DB is now ready ---"
