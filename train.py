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

def get_dataloaders(root='/home/gwidon/Documents/ZPO/data/malaria_dataset', batch_size: int = 32, num_workers: int = 4):   
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
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Slight brightness and contrast changes
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        AutoCrop(threshold=5),
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
    resnet18 = models.resnet18(weights="IMAGENET1K_V1")
    # Freeze all the layers except the final layer
    for param in resnet18.parameters():
        param.requires_grad = False
    num_classes = 2
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128),      # Step 1: reduce features to 128
    nn.ReLU(),                     # Step 2: Activation function (adds non-linearity)
    nn.Dropout(0.5),               # Step 3: Randomly drop 50% of neurons (prevents overfitting)
    nn.Linear(128, num_classes)    # Step 4: Final output - 1 number (will decide 0 or 1)
)


    # Initialize the model and trainer
    model = LitResNet(resnet18)

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
    best_model = LitResNet.load_from_checkpoint(best_model_path, model=resnet18)
    # Save the entire model
    torch.save(best_model.model, 'models/best.pt')

if __name__ == "__main__":
    main()