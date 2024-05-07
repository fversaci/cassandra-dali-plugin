#! /bin/bash -x
set -e

pgrep -f cassandra || /cassandra/bin/cassandra 2>&1 | grep "state jump to NORMAL" 
SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -e "DROP KEYSPACE IF EXISTS imagenette;" 
SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -f create_tables.cql 
python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=train --data-table=imagenette.metadata_train_256_jpg --metadata-table=imagenette.metadata_train_256_jpg 
python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=val --data-table=imagenette.metadata_val_256_jpg --metadata-table=imagenette.metadata_val_256_jpg
rm -f ids_cache/* 
python3 cache_uuids.py --metadata-table=imagenette.metadata_train_256_jpg 
python3 cache_uuids.py --metadata-table=imagenette.metadata_val_256_jpg 
python3 create-json.py
echo "--- DB is now ready ---"
