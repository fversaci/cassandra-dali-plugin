#!/usr/bin/env fish


function usage
    echo "Usage: script_name.fish --host <host> --bs <batch size> --epochs <num_epochs> --rootdir <root data dir> --logdir <log dir> --debug"
end

# Parse arguments
argparse 'h/host=' 'bs=' 'e/epochs=' 'd/rootdir=' 'logdir=' 'debug' -- $argv

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
    set _flag_epochs 10
end

# Access the values passed to the named parameters
set HOST $_flag_host
set BS $_flag_bs
set EPOCHS $_flag_epochs
set ROOT $_flag_rootdir
set LOG $_flag_logdir

# create log dir
mkdir -p $LOG

## Local filesystem
### files with DALI
echo "-- DALI FILES TEST --"
python3 loop_read.py --epochs $EPOCHS --bs $BS --reader file --file-root $ROOT/imagenet-files/train/ --log-fn "$LOG/$HOST"_loop_read_DALI_file_BS_"$BS"
	
### TFRecords with DALI
echo "-- DALI TFRECORDS TEST --"
python3 loop_read.py --epochs $EPOCHS --bs $BS --reader tfrecord --file-root $ROOT/imagenet-tfrecords/train/ --index-root $ROOT/imagenet-tfrecords/train_idx/ --log-fn "$LOG/$HOST"_loop_read_DALI_tfrecord_BS_"$BS"

### files with Pytorch
echo "-- PYTORCH FILES TEST --"
python3 pytorch_loop_read.py --epochs $EPOCHS --bs $BS --root-dir $ROOT/imagenet-files/train/ --log-fn "$LOG/$HOST"_loop_read_pytorch_files_BS_"$BS"
