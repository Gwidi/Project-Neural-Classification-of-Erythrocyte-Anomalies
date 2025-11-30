import torchvision.transforms as transforms
from dataset import MalariaDataset
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size: int = 32, num_workers: int = 4):   
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the malaria dataset
    trainval_dataset = MalariaDataset(split='trainval', transform=transform)
    test_dataset = MalariaDataset(split='test', transform=transform)

    # 1. Split train dataset into train and validation
    train, val = random_split(trainval_dataset, [0.75, 0.25])

    # 2. Create data loaders
    train_dataloader = DataLoader(train, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = DataLoader(val, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader