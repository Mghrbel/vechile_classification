import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import train_test_split


def images_transforms(IMAGE_SIZE=(300, 300)):
    data_transformation = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])]) 
    return data_transformation


def get_train_val_dataloader(path='data/train/augmented_data', batch_size=16, IMAGE_SIZE = (300, 300)) :
    
    train_val_dataset = datasets.ImageFolder(path, transform=images_transforms(IMAGE_SIZE))
    targets = train_val_dataset.targets

    train_idx, valid_idx= train_test_split(np.arange(len(targets)), test_size=0.1, shuffle=True, stratify=targets)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_val_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(train_val_dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, val_loader


def get_test_dataloader(path='data/test', batch_size=16, IMAGE_SIZE = (300, 300)) :
    test_dataset = datasets.ImageFolder(path, transform=images_transforms(IMAGE_SIZE))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return test_loader, test_dataset.classes