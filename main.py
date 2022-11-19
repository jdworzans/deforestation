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
        self.accuracy = torchmetrics.Accuracy()
        self.f1_score = torchmetrics.F1Score(num_classes=3, average="macro")

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
        y = self.model(self.preprocess(img))
        return y

    def step(self, batch, batch_idx, prefix: str):
        img, y = batch
        preds = self(img)
        loss = F.cross_entropy(preds, y)
        self.log(f"{prefix}/loss", loss)
        self.calculate_metrics(preds, y, prefix)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")
    
    def predict_step(self, batch, batch_idx):
        img = batch
        preds = self(img)
        return batch_idx, preds.argmax(-1)

    def calculate_metrics(self, preds, y, prefix):
        self.accuracy(preds, y)
        self.f1_score(preds, y)
        self.log(f"{prefix}/acc_step", self.accuracy)
        self.log(f"{prefix}/f1_step", self.f1_score)

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
        filename="model-{epoch:02d}-{valid_loss:.2E}",
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
