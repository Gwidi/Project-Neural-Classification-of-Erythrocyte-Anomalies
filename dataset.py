import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import glob
from PIL import Image

def read_image(img_path):
    return Image.open(img_path).convert('RGB')


class MalariaDataset(Dataset):
    def __init__(self, root: str = './data/malaria_dataset', split: str = 'trainval', transform=None):
        """
        Args:
            root (string): Path to the main directory with subdirectories 'train' and 'test'
            transform (callable, optional): Optional transformations
        """
        self.root = root
        self.split = split
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        if self.split == 'trainval':
            train_dir = os.path.join(self.root, 'train')
            if os.path.exists(train_dir):
                positive_images = glob.glob(os.path.join(train_dir,'positive', '*.png'))
                negative_images = glob.glob(os.path.join(train_dir,'negative', '*.png'))
                self.image_paths.extend(positive_images)
                self.labels.extend([1] * len(positive_images))  
                self.image_paths.extend(negative_images)
                self.labels.extend([0] * len(negative_images))
        if self.split == 'test':
            test_dir = os.path.join(self.root, 'test')
            if os.path.exists(test_dir):
                test_images = glob.glob(os.path.join(test_dir, '*.png'))
                self.image_paths.extend(test_images)
                self.labels.extend([-1] * len(test_images)) # Unknown labels for test set
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = read_image(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label