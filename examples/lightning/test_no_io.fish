#!/usr/bin/env fish

function usage
    echo "Usage: script_name.fish --host <host> --bs <batch size> --epochs <num_epochs>"
end

# Parse arguments
argparse 'h/host=' 'b/bs=' 'e/epochs=' -- $argv

if not set -q _flag_host
    set _flag_host NONE
end

if not set -q _flag_bs
    set _flag_bs 512
end

if not set -q _flag_epochs
    set _flag_epochs 4
end

# Access the values passed to the named parameters
set HOST $_flag_host
set BS $_flag_bs
set EPOCHS $_flag_epochs
set MAX_GPUS (nvidia-smi -L | wc -l)

### NO Dataloader
python3 train_model_no_IO.py --epoch $EPOCHS -b $BS --no-io -g 1 --log-csv "$HOST_1_GPU_NO_IO_BS_$BS"
python3 train_model_no_IO.py --epoch $EPOCHS -b $BS --no-io -g $MAX_GPUS --log-csv "$HOST_$MAX_GPUS_GPU_NO_IO_BS_$BS"
