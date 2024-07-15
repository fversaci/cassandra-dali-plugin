#! /bin/bash -x
set -e

pgrep -f cassandra || /cassandra/bin/cassandra 2>&1 | grep "state jump to NORMAL" 
    cd examples/lightning 
    SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -e "DROP KEYSPACE IF EXISTS imagenette;" 
    SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -f create_tables.cql 
    python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=train --data-table imagenette.data_train_orig --metadata-table imagenette.metadata_train_orig 
    python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=val --data-table imagenette.data_val_orig --metadata-table imagenette.metadata_val_orig 
    rm -f ids_cache/* 
    python3 cache_uuids.py --metadata-table=imagenette.metadata_train_orig 
    python3 cache_uuids.py --metadata-table=imagenette.metadata_val_orig 
    python3 train_model.py --num-gpu 1 -a resnet50 --b 64 --workers 4 --lr=1.0e-3 --epochs 1 \
                           --train-data-table imagenette.data_train_orig --train-metadata-table imagenette.metadata_train_orig \
                           --val-data-table imagenette.data_val_orig --val-metadata-table imagenette.metadata_val_orig`
   ### BEGIN COMMENT \
    ### END COMMENT`
    echo "--- OK ---"
