#!/usr/bin/env fish

function usage
    echo "Usage: script_name.fish --host <host> --bs <batch size> --epochs <num_epochs> --rootdir <root data dir> --logdir <log dir> --debug"
end

# Parse arguments
argparse 'h/host=' 'b/bs=' 'e/epochs=' 'd/rootdir=' 'logdir=' 'debug' -- $argv

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
    set _flag_logdir "log"
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
set ROOT $_flag_rootdir
set LOG $_flag_logdir

echo "LOCAL TEST"
echo "ROOT: $ROOT"
echo "LOG: $LOG"

### Files
set TRAIN_FOLDER $ROOT/imagenet-files/train
set VAL_FOLDER  $ROOT/imagenet-files/val

#python3 train_model.py --epoch $EPOCHS --train-folder "$TRAIN_FOLDER" --val-folder $VAL_FOLDER -g 1 -b $BS --log-csv "$LOG/$HOST"_1_GPU_FILE_BS_"$BS"

python3 train_model.py --epoch $EPOCHS --train-folder "$TRAIN_FOLDER" --val-folder $VAL_FOLDER -g $MAX_GPUS -b $BS --log-csv "$LOG/$HOST"_"$MAX_GPUS"_GPU_FILE_BS_"$BS"

### Tfrecords
set TRAIN_TFR $ROOT/imagenet-tfrecords/train
set TRAIN_INDEX $ROOT/imagenet-tfrecords/train_idx/
set VAL_TFR $ROOT/imagenet-tfrecords/val
set VAL_INDEX $ROOT/imagenet-tfrecords/val_idx/

#python3 train_model.py --epoch $EPOCHS --train-tfr-folder $TRAIN_TFR --train-index-folder $TRAIN_INDEX --val-tfr-folder $VAL_TFR --val-index-folder $VAL_INDEX -g 1 -b $BS --log-csv "$LOG/$HOST"_1_GPU_TFR_BS_"$BS"

#python3 train_model.py --epoch $EPOCHS --train-tfr-folder $TRAIN_TFR --train-index-folder $TRAIN_INDEX --val-tfr-folder $VAL_TFR --val-index-folder $VAL_INDEX -g $MAX_GPUS -b $BS --log-csv "$LOG/$HOST"_"$MAX_GPUS"_GPU_TFR_BS_"$BS"

