"""
Data Loading and Preprocessing
Handles dataset loading for both:
- Model B's dataset (your custom malware dataset)
- Cross-domain dataset (e.g., Malimg)
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import os
from typing import Tuple, Optional
import config


# ============================================================
# DATA TRANSFORMS
# ============================================================
def get_train_transforms(augmentation=True):
    """
    Get training data transforms with optional augmentation
    """
    transform_list = [
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    ]
    
    if augmentation and config.TRAIN_AUGMENTATION:
        aug_config = config.AUGMENTATION_CONFIG
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=aug_config['horizontal_flip']),
            transforms.RandomRotation(degrees=aug_config['rotation_degrees']),
            transforms.ColorJitter(
                brightness=aug_config['color_jitter']['brightness'],
                contrast=aug_config['color_jitter']['contrast'],
                saturation=aug_config['color_jitter']['saturation'],
                hue=aug_config['color_jitter']['hue']
            ),
        ])
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    return transforms.Compose(transform_list)


def get_test_transforms():
    """
    Get test/validation data transforms (no augmentation)
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])


# ============================================================
# DATASET LOADERS
# ============================================================
def load_dataset_b(batch_size=32, num_workers=4, augmentation=True):
    """
    Load Model B's dataset (your custom malware dataset)
    Assumes ImageFolder structure: dataset/train/class1/, dataset/train/class2/, etc.
    
    Args:
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of worker processes for data loading
        augmentation (bool): Whether to apply data augmentation to training set
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform = get_train_transforms(augmentation=augmentation)
    test_transform = get_test_transforms()
    
    # Load datasets using ImageFolder
    train_dataset = datasets.ImageFolder(
        root=config.DATASET_B_TRAIN,
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=config.DATASET_B_VAL,
        transform=test_transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=config.DATASET_B_TEST,
        transform=test_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset B loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Classes: {train_dataset.classes}")
    
    return train_loader, val_loader, test_loader


def load_cross_domain_dataset(batch_size=32, num_workers=4, augmentation=True):
    """
    Load cross-domain dataset (e.g., Malimg)
    Used for training Model A' and Model A''
    
    Args:
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of worker processes
        augmentation (bool): Whether to apply data augmentation
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform = get_train_transforms(augmentation=augmentation)
    test_transform = get_test_transforms()
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=config.DATASET_CROSS_DOMAIN_TRAIN,
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=config.DATASET_CROSS_DOMAIN_VAL,
        transform=test_transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=config.DATASET_CROSS_DOMAIN_TEST,
        transform=test_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Cross-domain dataset loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Classes: {train_dataset.classes}")
    
    return train_loader, val_loader, test_loader


# ============================================================
# CUSTOM DATASET FOR GAN TRAINING
# ============================================================
class AdversarialDataset(Dataset):
    """
    Custom dataset for GAN training
    Returns (image, true_label, target_label) tuples
    """
    def __init__(self, image_folder, transform=None, num_classes=8):
        """
        Args:
            image_folder (str): Path to image folder (ImageFolder structure)
            transform: torchvision transforms to apply
            num_classes (int): Total number of classes
        """
        self.dataset = datasets.ImageFolder(root=image_folder, transform=transform)
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, true_label = self.dataset[idx]
        
        # Generate a random target label (different from true label)
        target_label = true_label
        while target_label == true_label:
            target_label = torch.randint(0, self.num_classes, (1,)).item()
        
        return image, true_label, target_label


def load_gan_dataset(dataset_path, batch_size=16, num_workers=4):
    """
    Load dataset for GAN training
    
    Args:
        dataset_path (str): Path to dataset folder
        batch_size (int): Batch size
        num_workers (int): Number of workers
    
    Returns:
        DataLoader with (image, true_label, target_label) tuples
    """
    transform = get_test_transforms()  # No augmentation for GAN training
    
    dataset = AdversarialDataset(
        image_folder=dataset_path,
        transform=transform,
        num_classes=config.NUM_CLASSES
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"GAN training dataset loaded:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    
    return dataloader


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def denormalize(tensor, mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD):
    """
    Denormalize a tensor image for visualization
    
    Args:
        tensor: Normalized tensor (C, H, W) or (B, C, H, W)
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.dim() == 4:  # Batch of images
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    return tensor * std + mean


def save_image(tensor, filepath):
    """
    Save a tensor as an image
    
    Args:
        tensor: Image tensor (C, H, W) - assumes denormalized
        filepath: Path to save the image
    """
    from torchvision.utils import save_image as tv_save_image
    
    # Ensure tensor is in [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)
    tv_save_image(tensor, filepath)


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    print("Testing data loading utilities...")
    
    # Note: This will fail if paths are not set in config.py
    # Just for demonstrating the API
    try:
        print("\n1. Testing Dataset B loading...")
        train_loader, val_loader, test_loader = load_dataset_b(batch_size=8)
        
        # Get a batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        
    except Exception as e:
        print(f"Dataset B loading failed (expected if paths not configured): {e}")
    
    try:
        print("\n2. Testing cross-domain dataset loading...")
        train_loader, val_loader, test_loader = load_cross_domain_dataset(batch_size=8)
        
    except Exception as e:
        print(f"Cross-domain dataset loading failed (expected if paths not configured): {e}")
    
    print("\n3. Testing denormalization...")
    dummy_tensor = torch.randn(3, 224, 224)
    denorm_tensor = denormalize(dummy_tensor)
    print(f"Denormalized tensor shape: {denorm_tensor.shape}")
    
    print("\nData loading utilities tested successfully!")