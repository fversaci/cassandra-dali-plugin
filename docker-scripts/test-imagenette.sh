#! /bin/bash -x

pgrep -f cassandra || /cassandra/bin/cassandra 2>&1 | grep "state jump to NORMAL" && \
    cd examples/imagenette/ && \
    (/cassandra/bin/cqlsh -e "SELECT keyspace_name FROM system_schema.keyspaces WHERE keyspace_name='imagenette';" && \
         /cassandra/bin/cqlsh -e "DROP KEYSPACE imagenette;") || true && \
    /cassandra/bin/cqlsh -f create_tables.cql && \
    pgrep -f spark || (/spark/sbin/start-master.sh  && /spark/sbin/start-worker.sh spark://$HOSTNAME:7077) && \
    /spark/bin/spark-submit --master spark://$HOSTNAME:7077 --conf spark.default.parallelism=10 \
                            --py-files extract_common.py extract_spark.py /tmp/imagenette2-320/ \
                            --split-subdir=train --table-suffix=train_256_jpg && \
    /spark/bin/spark-submit --master spark://$HOSTNAME:7077 --conf spark.default.parallelism=10 \
                            --py-files extract_common.py extract_spark.py /tmp/imagenette2-320/ \
                            --split-subdir=val --table-suffix=val_256_jpg && \
    rm -f ids_cache/* && \
    python3 cache_uuids.py --table-suffix=train_256_jpg && \
    python3 loop_read.py --table-suffix=train_256_jpg && \
    python3 cache_uuids.py --table-suffix=val_256_jpg && \
    python3 loop_read.py --table-suffix=val_256_jpg && \
    python3 loop_read.py --table-suffix=train_256_jpg --device-id=0 && \
    /cassandra/bin/cqlsh -e "DROP KEYSPACE imagenette;" && \
    /cassandra/bin/cqlsh -f create_tables.cql && \
    python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=train --table-suffix=train_256_jpg && \
    python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=val --table-suffix=val_256_jpg && \
    rm -f ids_cache/* && \
    python3 cache_uuids.py --table-suffix=train_256_jpg && \
    python3 loop_read.py --table-suffix=train_256_jpg && \
    python3 cache_uuids.py --table-suffix=val_256_jpg && \
    python3 loop_read.py --table-suffix=val_256_jpg && \
    rm -Rf /tmp/imagenette_256_jpg/ && \
    python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=train --target-dir=/tmp/imagenette_256_jpg/train && \
    python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=val --target-dir=/tmp/imagenette_256_jpg/val && \
    python3 loop_read.py --reader=file --file-root=/tmp/imagenette_256_jpg/train && \
    `### BEGIN COMMENT \
    ### END COMMENT`
    echo "--- OK ---"
