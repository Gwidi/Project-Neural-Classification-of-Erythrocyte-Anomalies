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

def get_dataloaders(root='/home/gwidon/Documents/ZPO/data/malaria_dataset', batch_size: int = 32, num_workers: int = 4):   
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the malaria dataset
    trainval_dataset = MalariaDataset(split='trainval', transform=transform)

    # 1. Split train dataset into train validation and test sets
    total_size = len(trainval_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        trainval_dataset, [train_size, val_size, test_size])

    # 2. Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader



def main():
    train_loader, val_loader, test_loader = get_dataloaders()
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
        mode='min'                    # 'min' (loss), 'max' (accuracy)
    )
    trainer = L.Trainer(max_epochs=10, accelerator='gpu', logger=wandb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, test_loader)

if __name__ == "__main__":
    main()