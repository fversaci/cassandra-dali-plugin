#! /bin/bash -x
set -e

pgrep -f cassandra || /cassandra/bin/cassandra 2>&1 | grep "state jump to NORMAL"
cd examples/ade20k/
SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -e "DROP KEYSPACE IF EXISTS ade20k;"
SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -f create_tables.cql
pgrep -f spark || (/spark/sbin/start-master.sh  && /spark/sbin/start-worker.sh spark://$HOSTNAME:7077)
/spark/bin/spark-submit --master spark://$HOSTNAME:7077 --conf spark.default.parallelism=10 \
                        --py-files extract_common.py extract_spark.py /data/ade20k/images/training/ /data/ade20k/annotations/training/ \
                        --data-table=ade20k.data --metadata-table=ade20k.metadata
rm -f ade20k.rows
python3 cache_uuids.py --metadata-table=ade20k.metadata --rows-fn=ade20k.rows
python3 loop_read.py --data-table=ade20k.data --rows-fn=ade20k.rows
python3 loop_read.py --data-table=ade20k.data --rows-fn=ade20k.rows --use-gpu
SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -e "DROP KEYSPACE IF EXISTS ade20k;"
SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -f create_tables.cql
python3 extract_serial.py /data/ade20k/images/training/ /data/ade20k/annotations/training/ --data-table=ade20k.data --metadata-table=ade20k.metadata
rm -f ade20k.rows
python3 cache_uuids.py --metadata-table=ade20k.metadata --rows-fn=ade20k.rows
python3 loop_read.py --data-table=ade20k.data --rows-fn=ade20k.rows
python3 loop_read.py --reader=file --image-root=/data/ade20k/images/ --mask-root=/data/ade20k/annotations/
python3 loop_read.py --reader=file --image-root=/data/ade20k/images/ --mask-root=/data/ade20k/annotations/ --use-gpu
echo "--- OK ---"
