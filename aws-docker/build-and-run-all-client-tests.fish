#! /usr/bin/env fish

function usage
    echo "Usage: script_name.fish --host <host> --bs <batch size> --epochs <num_epochs> --rootdir <root data dir> --logdir <log dir> --iptf <ip_tf_server> --iptorch <ip_cass_server> --debug"
    end

# Parse arguments
argparse 'h/host=' 'bs=' 'e/epochs=' 'd/rootdir=' 'logdir=' 'iptf=' 'iptorch=' -- $argv

if set -q _flag_debug
    set fish_trace 1
end

if not set -q _flag_host
    set _flag_host NONE
end

if not set -q _flag_rootdir
    echo "rootdir is mandatory"
    exit
end

if not set -q _flag_logdir
    echo "logdir is mandatory"
    exit
end

if not set -q _flag_bs
    set _flag_bs 1024
end

if not set -q _flag_epochs
    set _flag_epochs 4
end

if not set -q _flag_iptf
    set _flag_ip 127.0.0.1
end

if not set -q _flag_iptorch
    set _flag_ip 127.0.0.1
end

# Access the values passed to the named parameters
set HOST $_flag_host
set BS $_flag_bs
set EPOCHS $_flag_epochs
set ROOT $_flag_rootdir
set LOG $_flag_logdir
set IPTF $_flag_iptf
set IPTORCH $_flag_iptorch

## Tensorflow tests
echo "--------------"
echo "-- TF tests --"
echo "--------------"

set start_time (date "+%Y-%m-%d %H:%M:%S")

cd ./tensorflow
./build-and-run-tensorflow-tests.fish --host $HOST --bs $BS --epochs $EPOCHS --rootdir $ROOT --logdir $LOG --ip $IPTF

set stop_tf_time (date "+%Y-%m-%d %H:%M:%S")

echo
echo
echo 
echo "-----------------"
echo "-- TORCH tests --"
echo "-----------------"

cd ../torch-client
./build-and-run-dali-torch-tests.fish --host $HOST --bs=1024 --epochs=4 --ip $IPTORCH --rootdir $ROOT --logdir $LOG

set stop_torch_time (date "+%Y-%m-%d %H:%M:%S")

echo "Start time: $start_time"
echo "Stop time tensorflow tests: $stop_tf_time"
echo "Stop time torch tests: $stop_torch_time"
