import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics import MetricCollection

class LitResNet(L.LightningModule):
    def __init__(self, model, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        # Use pre-trained ResNet18 model
        self.model = model
        
        self.loss = nn.CrossEntropyLoss()

        self.metrics = MetricCollection([
            torchmetrics.Accuracy(task="binary"),
            torchmetrics.F1Score(task="binary"),
            torchmetrics.Precision(task="binary"),
            torchmetrics.Recall(task="binary")
        ])
        self.train_metrics = self.metrics.clone(prefix='train_')
        self.val_metrics = self.metrics.clone(prefix='val_')
        self.test_metrics = self.metrics.clone(prefix='test_')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False)

        # convert logits to predicted class indices for metrics
        preds = torch.argmax(y_hat, dim=1)

        self.train_metrics.update(preds, y)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)


        # convert logits to predicted class indices for metrics
        preds = torch.argmax(y_hat, dim=1)

        self.val_metrics.update(preds, y)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        # convert logits to predicted class indices for metrics
        preds = torch.argmax(y_hat, dim=1)

        self.test_metrics.update(preds, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_loss",
        }
    }