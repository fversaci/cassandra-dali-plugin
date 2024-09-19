import os
import statistics
from clize import run
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm, trange


class ImageNetDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.files = [
            (os.path.join(dp, f), dp.split("/")[-1])
            for dp, dn, filenames in os.walk(root_dir)
            for f in filenames
            if f.endswith(".JPEG")
        ]
        self.transform = transforms.Lambda(
            lambda img: img
        )  # No transform, just return raw image

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, label = self.files[idx]
        with open(img_path, "rb") as f:
            img = f.read()  # Read raw image bytes
        label = self.classes.index(label)
        return img, label


def scan(*, root_dir="/data/imagenet/train", epochs=4):
    dataset = ImageNetDataset(root_dir)
    data_loader = DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=2, prefetch_factor=8
    )

    speeds = []
    first_epoch = True
    for _ in range(epochs):
        # read data for current epoch
        with tqdm(data_loader) as t:
            for _ in t:
                ...
            epoch_time = t.format_dict["elapsed"]
            if first_epoch:
                # ignore first epoch for stats
                first_epoch = False
            else:
                speeds.append(t.total / epoch_time)

    # Calculate the average and standard deviation
    if epochs > 3:
        average_speed = statistics.mean(speeds)
        std_dev_speed = statistics.stdev(speeds)
        print(f"Stats for epochs > 1")
        print(f"  Average speed: {average_speed:.2f} ± {std_dev_speed:.2f} it/s")


if __name__ == "__main__":
    run(scan)
