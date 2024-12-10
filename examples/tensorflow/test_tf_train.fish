#!/usr/bin/env fish

function usage
    echo "Usage: script_name.fish --host <host> --bs <batch size> --epochs <num_epochs> --rootdir <root data dir> --ip <ip_server> --logdir <log dir> --debug"
end

# Parse arguments
argparse 'h/host=' 'bs=' 'e/epochs=' 'd/rootdir=' 'ip=' 'logdir=' 'debug' -- $argv

if set -q _flag_debug
    set fish_trace 1
end

if not set -q _flag_host
    set _flag_host NONE
end

if not set -q _flag_rootdir
    set _flag_rootdir "/data"
end

if not set -q _flag_ip
    set _flag_ip 172.19.179.86
end

if not set -q _flag_logdir
    set _flag_logdir "log"
end

if not set -q _flag_bs
    set _flag_bs 1024
end

if not set -q _flag_epochs
    set _flag_epochs 4
end

# Access the values passed to the named parameters
set HOST $_flag_host
set BS $_flag_bs
set EPOCHS $_flag_epochs
set ROOT $_flag_rootdir
set LOG $_flag_logdir
set IP $_flag_ip
set -e http_proxy
set -e https_proxy

# create log dir
mkdir -p "$LOG/train"

##############
## TRAINING ##
##############
echo "-----------"
echo "TF Training"
echo "-----------"

echo "Tensorflow tf data service remote test"
sed -i "s/10.12.0.2/$IP/g" mynet.py

timeout -s SIGTERM 45m python3 train.py --tfr --bs $BS --epochs $EPOCHS --log-fn "$LOG/train/$HOST"_Train_TF_tfdataservice_tfr_BS_"$BS"
