#! /bin/bash -x
set -e

cd examples/corel5k/
ssh root@cassandra 'SSL_VALIDATE=false /opt/cassandra/bin/cqlsh --ssl -e "DROP KEYSPACE IF EXISTS corel5k;"'
cat create_tables.cql | ssh root@cassandra 'SSL_VALIDATE=false /opt/cassandra/bin/cqlsh --ssl'
python3 extract_serial.py /data/Corel-5k/images/ /data/Corel-5k/npy_labs /data/Corel-5k/train.json --data-table corel5k.data --metadata-table corel5k.metadata
rm -f corel5k.rows
python3 cache_uuids.py --metadata-table corel5k.metadata --rows-fn corel5k.rows
python3 loop_read.py --data-table corel5k.data --rows-fn corel5k.rows --use-gpu
echo "--- OK ---"
