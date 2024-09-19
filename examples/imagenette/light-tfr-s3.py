import boto3
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as pl


class S3TFRecordDataset(Dataset):
    def __init__(self, s3_bucket, s3_key):
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.data = self._download_and_load_tfrecord()

    def _download_and_load_tfrecord(self):
        s3 = boto3.client("s3")
        s3.download_file(self.s3_bucket, self.s3_key, "/tmp/temp.tfrecord")

        raw_dataset = tf.data.TFRecordDataset("/tmp/temp.tfrecord")
        return [raw_record.numpy() for raw_record in raw_dataset]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_record = self.data[idx]
        # Parse the raw record into features and labels
        # This part will depend on your specific TFRecord format
        feature_description = {
            "feature1": tf.io.FixedLenFeature([], tf.float32),
            "feature2": tf.io.FixedLenFeature([], tf.float32),
            # Add more features here
        }
        example = tf.io.parse_single_example(raw_record, feature_description)
        feature1 = example["feature1"].numpy()
        feature2 = example["feature2"].numpy()
        return torch.tensor(feature1), torch.tensor(feature2)


class MyDataModule(pl.LightningDataModule):
    def __init__(self, s3_bucket, s3_key, batch_size=32):
        super().__init__()
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = S3TFRecordDataset(self.s3_bucket, self.s3_key)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)


class MyModel(pl.LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = torch.nn.Linear(2, 1)  # Example model

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# Usage
s3_bucket = "imagenette"
s3_key = "tfrecords/train/train_0.tfrecord"

data_module = MyDataModule(s3_bucket, s3_key)
model = MyModel()

trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, data_module)
