import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Subset

def get_cifar10_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            # Normalize with CIFAR-10 mean and std
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
        
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_dataloaders(batch_size=64, root='./data'):
    """
    Returns standard CIFAR-10 loaders (all classes).
    """
    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=get_cifar10_transforms(train=True))
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=get_cifar10_transforms(train=False))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def get_cat_dog_dataloaders(batch_size=64, root='./data', fraction=1.0):
    """
    Returns loaders for only Cat (3) and Dog (5) classes.
    fraction: Float between 0 and 1. Fraction of the training data to use.
    """
    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=get_cifar10_transforms(train=True))
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=get_cifar10_transforms(train=False))
    
    # Filter for Cats (3) and Dogs (5)
    train_indices = [i for i, label in enumerate(train_dataset.targets) if label in [3, 5]]
    test_indices = [i for i, label in enumerate(test_dataset.targets) if label in [3, 5]]
    
    if fraction < 1.0:
        # Take a random subset of the filtered indices
        # Ensure we use a fixed seed for reproducibility of the subset if called multiple times, 
        # but here we just take the first N (shuffled by dataset usually or just by index).
        # To be safe and random:
        np.random.seed(42)
        train_indices = np.random.choice(train_indices, int(len(train_indices) * fraction), replace=False)
        
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def get_pretrain_dataloaders(batch_size=64, root='./data'):
    """
    Returns loaders for all classes EXCEPT Cat (3) and Dog (5).
    """
    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=get_cifar10_transforms(train=True))
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=get_cifar10_transforms(train=False))
    
    # Filter OUT Cats (3) and Dogs (5)
    train_indices = [i for i, label in enumerate(train_dataset.targets) if label not in [3, 5]]
    test_indices = [i for i, label in enumerate(test_dataset.targets) if label not in [3, 5]]
    
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader
