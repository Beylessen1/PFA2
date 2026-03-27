"""
Surrogate Model Training Script
Complete training pipeline for Model A, A', and A''
Includes training loop, validation, early stopping, and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import time
from tqdm import tqdm
import numpy as np

import config
from surrogate_models import ModelA_VGG16, ModelAPrime_ResNet50, ModelADoublePrime_EfficientNet
from data_utils import load_dataset_b, load_cross_domain_dataset


# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train model for one epoch
    
    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        Average training loss and accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch=None):
    """
    Validate model
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device
        epoch: Current epoch (for display)
    
    Returns:
        Average validation loss and accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    desc = f"Epoch {epoch} [Val]" if epoch is not None else "Validation"
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=desc)
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def train_surrogate_model(
    model,
    train_loader,
    val_loader,
    save_path,
    num_epochs=50,
    learning_rate=0.001,
    device='cuda'
):
    """
    Complete training pipeline for a surrogate model
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        save_path: Path to save the best model checkpoint
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
    
    Returns:
        Trained model
    """
    print(f"\n{'='*60}")
    print(f"Training Surrogate Model")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}\n")
    
    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.SURROGATE_CONFIG['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = StepLR(
        optimizer,
        step_size=config.SURROGATE_CONFIG['scheduler_step_size'],
        gamma=config.SURROGATE_CONFIG['scheduler_gamma']
    )
    
    # Early stopping
    best_val_acc = 0.0
    patience_counter = 0
    patience = config.SURROGATE_CONFIG['early_stopping_patience']
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Create checkpoint directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'history': history
            }
            torch.save(checkpoint, save_path)
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            break
        
        print("-" * 60)
    
    # Training complete
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Total Time: {elapsed_time/60:.2f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {save_path}")
    print(f"{'='*60}\n")
    
    # Load best model
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history


# ============================================================
# MAIN TRAINING FUNCTIONS FOR EACH MODEL
# ============================================================
def train_model_a():
    """
    Train Model A (Cross-Architecture: VGG16)
    Same dataset as Model B, different architecture
    """
    print("\n" + "="*80)
    print("TRAINING MODEL A: Cross-Architecture (VGG16)")
    print("Dataset: Same as Model B (your custom malware dataset)")
    print("="*80)
    
    # Load data
    train_loader, val_loader, test_loader = load_dataset_b(
        batch_size=config.SURROGATE_CONFIG['batch_size'],
        num_workers=config.SURROGATE_CONFIG['num_workers']
    )
    
    # Initialize model
    model = ModelA_VGG16(num_classes=config.NUM_CLASSES, pretrained=True)
    
    # Train
    trained_model, history = train_surrogate_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_path=config.MODEL_A_CHECKPOINT,
        num_epochs=config.SURROGATE_CONFIG['num_epochs'],
        learning_rate=config.SURROGATE_CONFIG['learning_rate'],
        device=config.DEVICE
    )
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = validate(trained_model, test_loader, criterion, config.DEVICE)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    return trained_model, history


def train_model_a_prime():
    """
    Train Model A' (Cross-Domain: ResNet-50)
    Same architecture as Model B, different dataset
    """
    print("\n" + "="*80)
    print("TRAINING MODEL A': Cross-Domain (ResNet-50)")
    print("Dataset: Different from Model B (e.g., Malimg)")
    print("="*80)
    
    # Load cross-domain data
    train_loader, val_loader, test_loader = load_cross_domain_dataset(
        batch_size=config.SURROGATE_CONFIG['batch_size'],
        num_workers=config.SURROGATE_CONFIG['num_workers']
    )
    
    # Initialize model
    model = ModelAPrime_ResNet50(num_classes=config.NUM_CLASSES, pretrained=True)
    
    # Train
    trained_model, history = train_surrogate_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_path=config.MODEL_A_PRIME_CHECKPOINT,
        num_epochs=config.SURROGATE_CONFIG['num_epochs'],
        learning_rate=config.SURROGATE_CONFIG['learning_rate'],
        device=config.DEVICE
    )
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = validate(trained_model, test_loader, criterion, config.DEVICE)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    return trained_model, history


def train_model_a_double_prime():
    """
    Train Model A'' (Cross-Architecture + Cross-Domain: EfficientNet)
    Different architecture AND different dataset
    """
    print("\n" + "="*80)
    print("TRAINING MODEL A'': Cross-Architecture + Cross-Domain (EfficientNet-B0)")
    print("Dataset: Different from Model B (e.g., Malimg)")
    print("="*80)
    
    # Load cross-domain data
    train_loader, val_loader, test_loader = load_cross_domain_dataset(
        batch_size=config.SURROGATE_CONFIG['batch_size'],
        num_workers=config.SURROGATE_CONFIG['num_workers']
    )
    
    # Initialize model
    model = ModelADoublePrime_EfficientNet(num_classes=config.NUM_CLASSES, pretrained=True)
    
    # Train
    trained_model, history = train_surrogate_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_path=config.MODEL_A_DOUBLE_PRIME_CHECKPOINT,
        num_epochs=config.SURROGATE_CONFIG['num_epochs'],
        learning_rate=config.SURROGATE_CONFIG['learning_rate'],
        device=config.DEVICE
    )
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = validate(trained_model, test_loader, criterion, config.DEVICE)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    return trained_model, history


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train surrogate models")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['a', 'a_prime', 'a_double_prime', 'all'],
        help="Which model to train: 'a', 'a_prime', 'a_double_prime', or 'all'"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    if args.model == 'a' or args.model == 'all':
        train_model_a()
    
    if args.model == 'a_prime' or args.model == 'all':
        train_model_a_prime()
    
    if args.model == 'a_double_prime' or args.model == 'all':
        train_model_a_double_prime()
    
    print("\n" + "="*80)
    print("ALL TRAINING COMPLETE!")
    print("="*80)