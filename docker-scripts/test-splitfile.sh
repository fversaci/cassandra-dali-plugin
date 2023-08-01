#! /bin/bash -x
set -e

pgrep -f cassandra || /cassandra/bin/cassandra 2>&1 | grep "state jump to NORMAL" 
    cd examples/splitfile/ 
    SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -e "DROP KEYSPACE IF EXISTS imagenette;" 
    SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -f create_tables.cql 
    python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=train --img-format=UNCHANGED --table-suffix=orig 
    python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=val --img-format=UNCHANGED --table-suffix=orig 
    rm -f imagenette_splitfile.pckl metadata.cache 
    python3 create_split.py -k imagenette -s orig -r 8,2 --metadata-ofn metadata.cache -o imagenette_splitfile.pckl 
    python3 loop_read.py imagenette_splitfile.pckl 
    python3 loop_read.py imagenette_splitfile.pckl --use-index=1 
    python3 create_split.py --metadata-ifn metadata.cache -r 8,2 -o imagenette_splitfile.pckl 
    python3 loop_read.py imagenette_splitfile.pckl 
    python3 loop_read.py imagenette_splitfile.pckl --use-index=1 
    torchrun --nproc_per_node=1 distrib_train_from_cassandra.py --split-fn imagenette_splitfile.pckl --train-index 1 \
      --val-index 0 -a resnet50 --dali_cpu --b 128 --loss-scale 128.0 --workers 4 --lr=0.4 --opt-level O2 --epochs 1 
    rm -f imagenette_splitfile.pckl metadata.cache 
    `### BEGIN COMMENT \
    ### END COMMENT`
    echo "--- OK ---"
