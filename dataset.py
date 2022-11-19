from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split


class DeforestationDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        data_dir: str,
        transform: Optional[Callable] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.data_dir = Path(data_dir)
        self.df = pd.read_csv(self.csv_path)
        self.verbose = verbose
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        filepath = self.data_dir / data["example_path"]
        img = torchvision.io.read_image(str(filepath))
        if self.transform is not None:
            img = self.transform(img)

        if "label" in data:
            return img, data["label"]
        else:
            return img

class DataModule(pl.LightningDataModule):
    def __init__(self, transform: Optional[Callable] = None, batch_size: int = 32):
        super().__init__()
        self.train_dataset = DeforestationDataset("data/split/train.csv", "data", transform)
        self.test = DeforestationDataset("data/split/test.csv", "data", transform)
        self.to_predict = DeforestationDataset("data/test.csv", "data", transform)

        train_size = int(0.8 * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size

        self.train, self.val = random_split(self.train_dataset, [train_size, val_size])
        self.batch_size = batch_size
        self.num_workers = 4
        print("Created DataModule")

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.to_predict, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    resnet_weights = torchvision.models.ResNet50_Weights.DEFAULT
    dataset = DeforestationDataset("data/train.csv", "data", resnet_weights.transforms(), verbose=True)
    example = dataset[0]
