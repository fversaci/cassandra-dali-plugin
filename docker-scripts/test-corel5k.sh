#! /bin/bash -x
set -e

pgrep -f cassandra || /cassandra/bin/cassandra 2>&1 | grep "state jump to NORMAL"
    cd examples/corel5k/
    SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -e "DROP KEYSPACE IF EXISTS corel5k;"
    SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -f create_tables.cql
    python3 extract_serial.py /data/Corel-5k/images/ /data/Corel-5k/npy_labs /data/Corel-5k/train.json --data-table corel5k.data --metadata-table corel5k.metadata
    rm -f ids_cache/*
    python3 cache_uuids.py --metadata-table corel5k.metadata
    python3 loop_read.py --data-table corel5k.data --metadata-table corel5k.metadata --use-gpu
    ### BEGIN COMMENT \
    ### END COMMENT`
    echo "--- OK ---"
