#!/usr/bin/env fish

function usage
    echo "Usage: script_name.fish --host <host> --bs <batch size> --epochs <num_epochs>"
end

# Parse arguments
argparse 'h/host=' 'b/bs=' 'g/gpus=' 'e/epochs=' -- $argv

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

echo $HOST

### Files
set TRAIN_FOLDER /scratch/imagenet-files/train
set VAL_FOLDER /scratch/imagenet-files/val

python3 train_model_no_IO.py --train-folder $TRAIN_FOLDER --val-folder $VAL_FOLDER -g 1 -b $BS --log-csv "{$HOST}_1_GPU_TFR_BS_{$BS}"

python3 train_model_no_IO.py --train-folder $TRAIN_FOLDER --val-folder $VAL_FOLDER -g $MAX_GPUS -b $BS --log-csv "{$HOST}_{$MAX_GPUS}_GPU_TFR_BS_{$BS}"

### Tfrecords
set TRAIN_TFR /scratch/imagenet-tfrecords/train
set TRAIN_INDEX /scratch/imagenet-tfrecords/train_idx/
set VAL_TFR /scratch/imagenet-tfrecords/val
set VAL_INDEX /scratch/imagenet-tfrecords/val_idx/

python3 train_model_no_IO.py --train-tfr-folder $TRAIN_TFR --train-index-folder $TRAIN_INDEX --val-tfr-folder $VAL_TFR --val-index-folder $VAL_INDEX -g 1 -b $BS --log-csv "{$HOST}_1_GPU_TFR_BS_{$BS}"

python3 train_model_no_IO.py --train-tfr-folder $TRAIN_TFR --train-index-folder $TRAIN_INDEX --val-tfr-folder $VAL_TFR --val-index-folder $VAL_INDEX -g $MAX_GPUS -b $BS --log-csv "{$HOST}_{$MAX_GPUS}_GPU_TFR_BS_{$BS}"

