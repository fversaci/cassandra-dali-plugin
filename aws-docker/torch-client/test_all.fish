#!/usr/bin/env fish

function usage
    echo "Usage: script_name.fish --host <host> --bs <batch size> --epochs <num_epochs> --ip <ip_server> --pc <port_cassandra> --ps <port_scylla> --rootdir <root data dir> --logdir <log dir> --debug"
end

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
mkdir -p "$LOG/torch"

## Loopread test
cd ~/cassandra-dali-plugin/examples/imagenette
source ./test_loop_read_all.fish --host $HOST --epochs $EPOCHS --bs $BS --rootdir $ROOT --ip $IP --logdir "$LOG/loopread"

## Training test
cd ~/cassandra-dali-plugin/examples/lightning
source ./test_all.fish --host $HOST --epochs $EPOCHS --bs $BS --rootdir $ROOT  --ip $IP --logdir "$LOG/training"
