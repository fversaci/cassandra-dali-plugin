#!/usr/bin/env fish

function usage
    echo "Usage: script_name.fish --host <host> --bs <batch size> --epochs <num_epochs> --ip <ip_server> --pc <port_cassandra> --ps <port_scylla> --rootdir <root data dir> --logdir <log dir> --debug"
end

set MAX_GPUS (nvidia-smi -L | wc -l)

# Parse arguments
argparse 'h/host=' 'bs=' 'e/epochs=' 'ip=' 'pc=' 'ps=' 'd/rootdir=' 'logdir=' 'debug' -- $argv

if set -q _flag_debug
    set fish_trace 1
end

if not set -q _flag_host
    set _flag_host NONE
end

if not set -q _flag_rootdir
    set _flag_rootdir "/data"
end

if not set -q _flag_logdir
    set _flag_logdir "/log"
end

if not set -q _flag_bs
    set _flag_bs 512
end

if not set -q _flag_epochs
    set _flag_epochs 10
end

if not set -q _flag_ip
    set _flag_ip 172.19.179.86
end

if not set -q _flag_pc
    set _flag_pc 9042
end

if not set -q _flag_ps
    set _flag_ps 9043
end

# Access the values passed to the named parameters
set HOST $_flag_host
set BS $_flag_bs
set EPOCHS $_flag_epochs
set ROOT $_flag_rootdir
set LOG $_flag_logdir
set IP $_flag_ip
set PORT_SCYLLA $_flag_ps
set PORT_CASS $_flag_pc

# create log dir
mkdir -p $LOG

echo "Host: $HOST"
echo "Batch Size: $BS"
echo "Max GPUs: $MAX_GPUS"
echo "Epochs: $EPOCHS"
echo "IP server: $IP"
echo "Cassandra port: $PORT_CASS"
echo "Scylla port: $PORT_SCYLLA"
echo "Root dir: $ROOT"
echo "Log dir: $LOG"

### Run tests

set start_time (date "+%Y-%m-%d %H:%M:%S")

echo "NO DATALOADER TEST"
source ./test_no_io.fish --host $HOST --bs $BS --epochs $EPOCHS --logdir $LOG
set no_io_test_end_time (date "+%Y-%m-%d %H:%M:%S")

echo "LOCAL FS TEST"
source ./test_local.fish --host $HOST --bs $BS --epochs $EPOCHS --rootdir $ROOT --logdir $LOG
set local_test_end_time (date "+%Y-%m-%d %H:%M:%S")

echo "HILAT TEST"
source ./test_hi_lat.fish --host $HOST --bs $BS --epochs $EPOCHS --ip $_flag_ip --pc $PORT_CASS --ps $PORT_SCYLLA --logdir $LOG
set hilat_test_end_time (date "+%Y-%m-%d %H:%M:%S")

echo "STREAMING TEST"
source ./test_streaming.fish --host $HOST --bs $BS --epochs $EPOCHS --ip $_flag_ip --logdir $LOG
set stop_time (date "+%Y-%m-%d %H:%M:%S")

echo "Start time: $start_time"
echo "no IO stop time: $no_io_test_end_time"
echo "local stop time: $local_test_end_time"
echo "hilat stop time: $hilat_test_end_time"
echo "streaming stop time: $stop_time"
