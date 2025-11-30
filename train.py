import lightning as L
import torch
import torch.nn as nn
import torchvision.models as models
from LitResNet import LitResNet
import torchvision.transforms as transforms
from dataset import MalariaDataset
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.loggers import MLFlowLogger

def get_dataloaders(root='../data/malaria_dataset/', batch_size: int = 32, num_workers: int = 4):   
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
    resnet18.fc = nn.Linear(num_ftrs, num_classes)

    # Initialize the model and trainer
    model = LitResNet(resnet18, num_classes=2)

    # Exercise 2 Train the model and verify its performance on the test set.
    experiment_name = "resnet18_finetune_malaria_dataset"
    run_name = "basic_finetuning"
    mlf_logger = MLFlowLogger(experiment_name=experiment_name, run_name=run_name, tracking_uri="file:./mlruns")
    trainer = L.Trainer(max_epochs=10, accelerator='gpu', logger=mlf_logger)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, test_loader)

if __name__ == "__main__":
    main()