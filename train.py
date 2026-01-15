import lightning as L
import torch
import torch.nn as nn
import torchvision.models as models
from LitResNet import LitResNet
import torchvision.transforms as transforms
from dataset import MalariaDataset
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from auto_crop import AutoCrop

class MixupCutmix: 
    def __init__(self, mixup_alpha=0.4, cutmix_alpha=1.0, prob=0.5):
        self.mixup_alpha = mixup_alpha
        self. cutmix_alpha = cutmix_alpha
        self. prob = prob
    
    def __call__(self, images, labels):
        if torch.rand(1) > self.prob:
            return images, labels, labels, 1.0
        
        batch_size = images. size(0)
        indices = torch.randperm(batch_size)
        
        if torch.rand(1) > 0.5:
            # Mixup
            lam = torch.distributions.Beta(self. mixup_alpha, self.mixup_alpha).sample()
            images = lam * images + (1 - lam) * images[indices]
        else:
            # CutMix
            lam = torch.distributions.Beta(self.cutmix_alpha, self. cutmix_alpha).sample()
            _, _, H, W = images. shape
            cut_ratio = torch.sqrt(1 - lam)
            cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
            cx, cy = torch.randint(W, (1,)), torch.randint(H, (1,))
            x1 = torch.clamp(cx - cut_w // 2, 0, W)
            x2 = torch. clamp(cx + cut_w // 2, 0, W)
            y1 = torch. clamp(cy - cut_h // 2, 0, H)
            y2 = torch. clamp(cy + cut_h // 2, 0, H)
            images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
            lam = 1 - ((x2 - x1) * (y2 - y1) / (H * W))
        
        return images, labels, labels[indices], lam. item()

def get_dataloaders(root='/home/gwidon/Documents/ZPO/data/malaria_dataset', batch_size: int = 128, num_workers: int = 32):   
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),      # 0 degrees
            transforms.RandomRotation((90, 90)),    # 90 degrees
            transforms.RandomRotation((180, 180)),  # 180 degrees
            transforms.RandomRotation((270, 270)),  # 270 degrees
        ]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.1),  # Slight brightness and contrast changes
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the malaria dataset
    trainval_dataset = MalariaDataset(split='trainval', transform=None)

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
        trainval_dataset, [0.8, 0.2], generator=generator)
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # 2. Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, val_dataloader



def main():
    train_loader, val_loader = get_dataloaders()
    # Load ResNet-18 model
    resnet50 = models.resnet50(weights="IMAGENET1K_V2")
    # Freeze all the layers except the final layer
    for param in resnet50.parameters():
        param.requires_grad = False
    num_classes = 2
    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),      # Step 1: reduce features to 128
    nn.ReLU(),                     # Step 2: Activation function (adds non-linearity)
    nn.Dropout(0.5),               # Step 3: Randomly drop 50% of neurons (prevents overfitting)
    nn.Linear(256, num_classes)    # Step 4: Final output - 1 number (will decide 0 or 1)
)
    # Mixup/CutMix
    mixup_cutmix = MixupCutmix(mixup_alpha=0.4, cutmix_alpha=1.0, prob=0.5)

    # Initialize the model and trainer
    model = LitResNet(resnet50, learning_rate=1e-3, unfreeze_epoch=5, unfreeze_layers=1, mixup_fn=mixup_cutmix)

    # Exercise 2 Train the model and verify its performance on the test set.
    experiment_name = "resnet18_transfer_learning"
    run_name = "basic_finetuning"
    wandb_logger = WandbLogger(project=experiment_name, name=run_name)
    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',           # Metric
        dirpath='models/',     
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,                 # Save only the best model
        mode='min',                   # 'min' (loss), 'max' (accuracy)
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = L.Trainer(max_epochs=100, accelerator='gpu', logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best_model_path = checkpoint_callback.best_model_path
    best_model = LitResNet.load_from_checkpoint(best_model_path, model=resnet50)
    # Save the entire model
    torch.save(best_model.model, 'models/best.pt')

if __name__ == "__main__":
    main()
