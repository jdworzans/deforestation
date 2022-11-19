import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningCLI
from torch import nn

from dataset import DataModule


class DeforestationModule(pl.LightningModule):
    def __init__(self, learning_rate: float = 0.0001):
        super().__init__()
        self.save_hyperparameters()
        self._setup_model()
        metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(), torchmetrics.F1Score(num_classes=3, average="macro")])
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

    def _setup_model(self):
        initial_weights = torchvision.models.VGG11_Weights.DEFAULT
        self.preprocess = initial_weights.transforms()
        self.model = torchvision.models.vgg11(weights=initial_weights)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.classifier[-1] = nn.Linear(4096, 3)
        for p in self.model.classifier[-1].parameters():
            p.requires_grad = True

    def forward(self, x):
        img = x
        y = self.model(img)
        return y

    def training_step(self, batch, batch_idx):
        img, y = batch
        preds = self(img)
        loss = F.cross_entropy(preds, y)
        self.log(f"train/loss", loss)
        self.log_dict(self.train_metrics(preds, y))
        return loss

    def validation_step(self, batch, batch_idx):
        img, y = batch
        preds = self(img)
        loss = F.cross_entropy(preds, y)
        self.log(f"val/loss", loss)
        self.val_metrics.update(preds, y)
        return loss

    def validation_epoch_end(self, outputs) -> None:
        self.log_dict(self.val_metrics.compute())

    def test_step(self, batch, batch_idx):
        img, y = batch
        preds = self(img)
        loss = F.cross_entropy(preds, y)
        self.log(f"test/loss", loss)
        self.log_dict(self.test_metrics.update(preds, y))
        return loss    

    def test_epoch_end(self, outputs) -> None:
        self.log_dict(self.test_metrics.compute())

    def predict_step(self, batch, batch_idx):
        img = batch
        preds = self(img)
        return batch_idx, preds.argmax(-1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

class DeforestationTrainer(pl.Trainer):
    def predict(self, *args, **kwargs):
        predictions = super().predict(*args, **kwargs)
        predictions = torch.cat([p for (idx, p) in sorted(predictions)])
        result = pd.DataFrame({"target": predictions})
        result.to_json("predictions.json", indent=True)


if __name__ == "__main__":
    pl.seed_everything(257)
    early_stop_callback = EarlyStopping(
        monitor="val/loss", min_delta=0.00, patience=10, verbose=False, mode="min"
    )
    # Helper callback for saving models
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        filename="model-{epoch:02d}-{val/loss:.2E}",
        save_top_k=3,
        mode="min",
    )

    cli = LightningCLI(
        DeforestationModule,
        DataModule,
        trainer_class=DeforestationTrainer,
        trainer_defaults={
            'gpus': 1,
            'callbacks': [early_stop_callback, checkpoint_callback],
        },
        seed_everything_default=1234,
        save_config_overwrite=True,
    )
