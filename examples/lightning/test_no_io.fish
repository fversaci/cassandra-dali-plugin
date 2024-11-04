#!/usr/bin/env fish

function usage
    echo "Usage: script_name.fish --host <host> --bs <batch size> --epochs <num_epochs> --logdir <log dir> --debug"
end

# Parse arguments
argparse 'h/host=' 'b/bs=' 'e/epochs=' 'logdir=' 'debug' -- $argv

if set -q _flag_debug
    set fish_trace 1
end

if not set -q _flag_host
    set _flag_host NONE
end

if not set -q _flag_bs
    set _flag_bs 512
end

if not set -q _flag_epochs
    set _flag_epochs 4
end

if not set -q _flag_logdir
    set _flag_logdir "log"
end

# Access the values passed to the named parameters
set HOST $_flag_host
set BS $_flag_bs
set EPOCHS $_flag_epochs
set MAX_GPUS (nvidia-smi -L | wc -l)

### NO Dataloader
python3 train_model.py --epoch $EPOCHS -b $BS --no-io -g 1 --log-csv "$LOG/$HOST"_1_GPU_NO_IO_BS_"$BS"
python3 train_model.py --epoch $EPOCHS -b $BS --no-io -g $MAX_GPUS --log-csv "$LOG/$HOST"_"$MAX_GPUS"_GPU_NO_IO_BS_"$BS"
