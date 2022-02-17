import pytorch_lightning as pl
from qpi_simulation import save_data, QPIDataSet
from torch.utils.data import DataLoader
import os


class QPIDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # called only on 1 GPU
        measurement_shape = (1, 128, 128)
        kernel_shape = (16, 16)

        save_data(20000, measurement_shape, kernel_shape, training=True)
        save_data(1000, measurement_shape, kernel_shape, validation=True)
        save_data(500, measurement_shape, kernel_shape, testing=True)

    def setup(self, stage=None):
        # called on every GPU
        self.train = QPIDataSet(os.getcwd() + '/training_dataset')
        self.val = QPIDataSet(os.getcwd() + '/validation_dataset')
        self.test = QPIDataSet(os.getcwd() + '/testing_dataset')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=0)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=0)