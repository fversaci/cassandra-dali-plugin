#! /bin/bash -x

pgrep -f cassandra || /cassandra/bin/cassandra 2>&1 | grep "state jump to NORMAL" && \
    cd examples/ade20k/ && \
    (/cassandra/bin/cqlsh -e "SELECT keyspace_name FROM system_schema.keyspaces WHERE keyspace_name='ade20k';" && \
         /cassandra/bin/cqlsh -e "DROP KEYSPACE ade20k;") || true && \
    /cassandra/bin/cqlsh -f create_tables.cql && \
    pgrep -f spark || (sudo /spark/sbin/start-master.sh  && sudo /spark/sbin/start-worker.sh spark://$HOSTNAME:7077) && \
    /spark/bin/spark-submit --master spark://$HOSTNAME:7077 --conf spark.default.parallelism=10 \
                            --py-files extract_common.py extract_spark.py /data/ade20k/images/training/ /data/ade20k/annotations/training/ \
                            --table-suffix=orig && \
    rm -f ids_cache/* && \
    python3 loop_read.py --table-suffix=orig && \
    python3 loop_read.py --table-suffix=orig --device-id=0 && \
    /cassandra/bin/cqlsh -e "DROP KEYSPACE ade20k;" && \
    /cassandra/bin/cqlsh -f create_tables.cql && \
    python3 extract_serial.py /data/ade20k/images/training/ /data/ade20k/annotations/training/ --table-suffix=orig && \
    rm -f ids_cache/* && \
    python3 loop_read.py --table-suffix=orig && \
    python3 loop_read.py --reader=file --image-root=/data/ade20k/images/ --mask-root=/data/ade20k/annotations/ && \
    python3 loop_read.py --reader=file --image-root=/data/ade20k/images/ --mask-root=/data/ade20k/annotations/ --device-id=0 && \
    `### BEGIN COMMENT \
    ### END COMMENT`
    echo "--- OK ---"
