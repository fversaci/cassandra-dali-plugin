from clize import run
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper
import statistics
from IPython import embed


def custom_collate(batch):
    return batch


def scan(*, root_dir="s3://imagenet/val", epochs=4):
    s3_shard_urls = IterableWrapper(
        [
            root_dir,
        ]
    ).list_files_by_s3()
    s3_files = s3_shard_urls.sharding_filter().load_files_by_s3()
    # tf_records = s3_files.load_from_tfrecord()
    # text data
    data_loader = DataLoader(
        s3_files,
        batch_size=128,
        num_workers=32,
        prefetch_factor=8,
        collate_fn=custom_collate,
        shuffle=True,
    )

    speeds = []
    first_epoch = True
    for _ in range(epochs):
        # read data for current epoch
        with tqdm(data_loader) as t:
            for b in t:
                pass
            epoch_time = t.format_dict["elapsed"]
            if first_epoch:
                # ignore first epoch for stats
                first_epoch = False
            else:
                speeds.append(t.n / epoch_time)

    # Calculate the average and standard deviation
    if epochs > 3:
        average_speed = statistics.mean(speeds)
        std_dev_speed = statistics.stdev(speeds)
        print(f"Stats for epochs > 1")
        print(f"  Average speed: {average_speed:.2f} ± {std_dev_speed:.2f} it/s")


if __name__ == "__main__":
    run(scan)
