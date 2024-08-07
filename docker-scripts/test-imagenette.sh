#! /bin/bash -x
set -e

# start cassandra and (re)create tables
pgrep -f cassandra || /cassandra/bin/cassandra 2>&1 | grep "state jump to NORMAL" 
cd examples/imagenette/ 
SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -e "DROP KEYSPACE IF EXISTS imagenette;" 
SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -f create_tables.cql 
# start spark and insert dataset
pgrep -f spark || (/spark/sbin/start-master.sh  && /spark/sbin/start-worker.sh spark://$HOSTNAME:7077) 
/spark/bin/spark-submit --master spark://$HOSTNAME:7077 --conf spark.default.parallelism=10 \
                        --py-files extract_common.py extract_spark.py /tmp/imagenette2-320/ \
                        --split-subdir=train --data-table=imagenette.data_train_orig \
    		    --metadata-table=imagenette.metadata_train_orig 
/spark/bin/spark-submit --master spark://$HOSTNAME:7077 --conf spark.default.parallelism=10 \
                        --py-files extract_common.py extract_spark.py /tmp/imagenette2-320/ \
                        --split-subdir=val --data-table=imagenette.data_val_orig \
    		    --metadata-table=imagenette.metadata_val_orig 
# read from db
rm -f ids_cache/* 
python3 cache_uuids.py --metadata-table=imagenette.metadata_train_orig 
python3 loop_read.py --data-table imagenette.data_train_orig --metadata-table imagenette.metadata_train_orig 
python3 cache_uuids.py --metadata-table=imagenette.metadata_val_orig 
python3 loop_read.py --data-table imagenette.data_val_orig --metadata-table imagenette.metadata_val_orig
python3 loop_read.py --data-table imagenette.data_train_orig --metadata-table imagenette.metadata_train_orig --use-gpu 
# recreate tables and insert serially
SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -e "DROP KEYSPACE IF EXISTS imagenette;" 
SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -f create_tables.cql
python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=train --data-table imagenette.data_train_orig --metadata-table imagenette.metadata_train_orig
python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=val --data-table imagenette.data_val_orig --metadata-table imagenette.metadata_val_orig
# read from db
rm -f ids_cache/* 
python3 cache_uuids.py --metadata-table=imagenette.metadata_train_orig 
python3 loop_read.py --data-table imagenette.data_train_orig --metadata-table imagenette.metadata_train_orig 
python3 cache_uuids.py --metadata-table=imagenette.metadata_val_orig 
python3 loop_read.py --data-table imagenette.data_val_orig --metadata-table imagenette.metadata_val_orig
# read from filesystem
python3 loop_read.py --reader=file --file-root=/tmp/imagenette2-320/train 
# train for one epoch
torchrun --nproc_per_node=1 distrib_train_from_cassandra.py -a resnet50 --dali_cpu --b 128 --loss-scale 128.0 \
         --workers 4 --lr=0.4 --opt-level O2 --epochs 1 \
         --train-data-table imagenette.data_train_orig --train-metadata-table imagenette.metadata_train_orig \
         --val-data-table imagenette.data_val_orig --val-metadata-table imagenette.metadata_val_orig
### BEGIN COMMENT \
### END COMMENT`
echo "--- OK ---"
