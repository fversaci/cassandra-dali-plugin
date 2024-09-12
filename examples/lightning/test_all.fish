#!/usr/bin/env fish

function usage
    echo "Usage: script_name.fish --host <host> --bs <batch size> --epochs <num_epochs> --ipc <ip_cassandra> --ips<ip_scylla>"
end

# Parse arguments
argparse 'h/host=' 'b/bs=' 'e/epochs=' 'ipc' 'ips' -- $argv

if not set -q _flag_host
    set _flag_host DOCKER
end

if not set -q _flag_bs
    set _flag_bs 512
end

if not set -q _flag_epochs
    set _flag_epochs 4
end

if not set -q _flag_ipc
    set _flag_ipc 172.19.179.85
end

if not set -q _flag_ips
    set _flag_ips 172.19.179.86
end

# Access the values passed to the named parameters
set HOST $_flag_host
set BS $_flag_bs
set EPOCHS $_flag_epochs
set MAX_GPUS (nvidia-smi -L | wc -l)

echo "Host: $HOST"
echo "Batch Size: $BS"
echo "Max GPUs: $MAX_GPUS"
echo "Epochs: $EPOCHS"
echo "IP Cassandra server: $_flag_ipc"
echo "IP Scylla server: $_flag_ips"


### Run tests
source ./test_no_io.fish --host $HOST --bs $BS --epochs $EPOCHS
source ./test_local.fish --host $HOST --bs $BS --epochs $EPOCHS
source ./test_hi_lat.fish --host $HOST --bs $BS --epochs $EPOCHS --ipc $_flag_ipc --ips $_flag_ips
