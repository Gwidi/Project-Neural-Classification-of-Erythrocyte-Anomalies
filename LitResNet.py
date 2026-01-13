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
    def __init__(self, model, learning_rate: float = 1e-3, unfreeze_epoch: int = 5, unfreeze_layers: int = 1):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        # Use pre-trained ResNet18 model
        self.model = model
        
        self.loss = nn.CrossEntropyLoss()

        # Select layers to unfreeze
        self.resnet_layers = [
            self.model.layer4,
            self.model. layer3,
            self.model. layer2,
            self.model.layer1
        ]
        
        self.unfreeze_epoch = unfreeze_epoch
        self.unfreeze_layers = unfreeze_layers
        self.unfrozen_count = 0

        self.metrics = MetricCollection([
            torchmetrics.Accuracy(task="binary"),
            torchmetrics.F1Score(task="binary"),
            torchmetrics.Precision(task="binary"),
            torchmetrics.Recall(task="binary")
        ])
        self.train_metrics = self.metrics.clone(prefix='train_')
        self.val_metrics = self.metrics.clone(prefix='val_')
        self.test_metrics = self.metrics.clone(prefix='test_')

    def on_train_epoch_start(self):
        """Callback to unfreeze layers at the start of each epoch."""
        current_epoch = self.current_epoch
        
        # Check if it's time to unfreeze layers
        if current_epoch > 0 and current_epoch % self.unfreeze_epoch == 0:
            layers_to_unfreeze = min(self.unfreeze_layers, len(self.resnet_layers) - self.unfrozen_count)
            
            for i in range(layers_to_unfreeze):
                if self.unfrozen_count < len(self.resnet_layers):
                    layer = self.resnet_layers[self. unfrozen_count]
                    for param in layer.parameters():
                        param.requires_grad = True
                    print(f"Epoch {current_epoch}:  Unfroze layer{4 - self.unfrozen_count}")
                    self. unfrozen_count += 1

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
        # Different learning rates for unfrozen layers (discriminative fine-tuning)
        param_groups = [
            {'params': self.model.fc. parameters(), 'lr': self.hparams.learning_rate},
        ]
        
        # Smaller learning rates for earlier layers
        for i, layer in enumerate(self.resnet_layers):
            param_groups.append({
                'params':  layer.parameters(),
                'lr': self.hparams. learning_rate / (10 ** (i + 1))  # Coraz mniejszy LR
            })
        optimizer = optim.AdamW(param_groups)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_loss",
        }
    }