#! /bin/bash -x

pgrep -f cassandra || /cassandra/bin/cassandra 2>&1 | grep "state jump to NORMAL" && \
    cd examples/ade20k/ && \
    (SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -e "SELECT keyspace_name FROM system_schema.keyspaces WHERE keyspace_name='ade20k';" && \
         SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -e "DROP KEYSPACE ade20k;") || true && \
    SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -f create_tables.cql && \
    pgrep -f spark || (/spark/sbin/start-master.sh  && /spark/sbin/start-worker.sh spark://$HOSTNAME:7077) && \
    /spark/bin/spark-submit --master spark://$HOSTNAME:7077 --conf spark.default.parallelism=10 \
                            --py-files extract_common.py extract_spark.py /data/ade20k/images/training/ /data/ade20k/annotations/training/ \
                            --table-suffix=orig && \
    rm -f ids_cache/* && \
    python3 cache_uuids.py --keyspace=ade20k --table-suffix=orig && \
    python3 loop_read.py --keyspace=ade20k --table-suffix=orig && \
    python3 loop_read.py --keyspace=ade20k --table-suffix=orig --use-gpu && \
    SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -e "DROP KEYSPACE ade20k;" && \
    SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -f create_tables.cql && \
    python3 extract_serial.py /data/ade20k/images/training/ /data/ade20k/annotations/training/ --table-suffix=orig && \
    rm -f ids_cache/* && \
    python3 cache_uuids.py --keyspace=ade20k --table-suffix=orig && \
    python3 loop_read.py --keyspace=ade20k --table-suffix=orig && \
    python3 loop_read.py --reader=file --image-root=/data/ade20k/images/ --mask-root=/data/ade20k/annotations/ && \
    python3 loop_read.py --reader=file --image-root=/data/ade20k/images/ --mask-root=/data/ade20k/annotations/ --use-gpu && \
    `### BEGIN COMMENT \
    ### END COMMENT`
    echo "--- OK ---"
