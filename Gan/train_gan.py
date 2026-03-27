"""
GAN Training Script for Adversarial Attack Generation
Complete training pipeline with all loss functions and training loop
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import numpy as np
import time

import config
from gan_models import Generator, Discriminator, weights_init
from surrogate_models import load_surrogate_model
from data_utils import load_gan_dataset, denormalize


# ============================================================
# LOSS FUNCTIONS
# ============================================================
class GANLosses:
    """
    Collection of loss functions for GAN-based adversarial attack
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def adversarial_loss(self, discriminator_output, is_real):
        """
        GAN adversarial loss (binary cross-entropy)
        
        Args:
            discriminator_output: Output from discriminator [B, 1]
            is_real: Boolean indicating if samples are real or fake
        
        Returns:
            loss: Adversarial loss value
        """
        if is_real:
            target = torch.ones_like(discriminator_output)
        else:
            target = torch.zeros_like(discriminator_output)
        
        return self.bce_loss(discriminator_output, target)
    
    def attack_loss(self, surrogate_model, adversarial_images, target_labels):
        """
        Attack loss: encourages adversarial images to be classified as target class
        
        Args:
            surrogate_model: The surrogate model to fool
            adversarial_images: Generated adversarial images [B, 3, H, W]
            target_labels: Target class labels [B]
        
        Returns:
            loss: Attack loss (cross-entropy)
        """
        outputs = surrogate_model(adversarial_images)
        return self.ce_loss(outputs, target_labels)
    
    def perturbation_loss(self, perturbation, norm='linf', epsilon=0.1):
        """
        Perturbation constraint loss
        Penalizes large perturbations
        
        Args:
            perturbation: Generated perturbation [B, 3, H, W]
            norm: 'l2' or 'linf'
            epsilon: Maximum allowed perturbation
        
        Returns:
            loss: Perturbation constraint loss
        """
        if norm == 'l2':
            # L2 norm: sqrt(sum of squares)
            per_sample_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=2, dim=1)
            return torch.mean(per_sample_norm)
        
        elif norm == 'linf':
            # L-infinity norm: maximum absolute value
            per_sample_norm = torch.max(torch.abs(perturbation.view(perturbation.size(0), -1)), dim=1)[0]
            # Penalize if exceeds epsilon
            return torch.mean(torch.relu(per_sample_norm - epsilon))
        
        else:
            raise ValueError(f"Unknown norm: {norm}")
    
    def feature_loss(self, surrogate_model, original_images, adversarial_images):
        """
        Feature-level loss: keeps adversarial images close to originals in feature space
        This helps maintain image structure while changing classification
        
        Args:
            surrogate_model: The surrogate model
            original_images: Original images [B, 3, H, W]
            adversarial_images: Adversarial images [B, 3, H, W]
        
        Returns:
            loss: Feature-level MSE loss
        """
        # Extract features from both original and adversarial images
        with torch.no_grad():
            original_features = surrogate_model.get_features(original_images)
        
        adversarial_features = surrogate_model.get_features(adversarial_images)
        
        # MSE between feature representations
        return self.mse_loss(adversarial_features, original_features)


# ============================================================
# GAN TRAINER CLASS
# ============================================================
class GANTrainer:
    """
    Complete GAN training pipeline for adversarial attack generation
    """
    
    def __init__(
        self,
        generator,
        discriminator,
        surrogate_model,
        device='cuda',
        config_dict=None
    ):
        """
        Args:
            generator: Generator network
            discriminator: Discriminator network
            surrogate_model: Surrogate model to attack (frozen)
            device: Training device
            config_dict: Configuration dictionary (from config.GAN_CONFIG)
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.surrogate_model = surrogate_model.to(device)
        self.device = device
        
        # Freeze surrogate model (we don't train it)
        self.surrogate_model.eval()
        for param in self.surrogate_model.parameters():
            param.requires_grad = False
        
        # Configuration
        if config_dict is None:
            config_dict = config.GAN_CONFIG
        self.config = config_dict
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=self.config['lr_generator'],
            betas=(self.config['beta1'], self.config['beta2'])
        )
        
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config['lr_discriminator'],
            betas=(self.config['beta1'], self.config['beta2'])
        )
        
        # Loss functions
        self.losses = GANLosses(device=device)
        
        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'attack_success_rate': []
        }
    
    def clamp_perturbation(self, perturbation, epsilon):
        """
        Clamp perturbation to be within [-epsilon, epsilon]
        
        Args:
            perturbation: Perturbation tensor [B, 3, H, W]
            epsilon: Maximum perturbation magnitude
        
        Returns:
            Clamped perturbation
        """
        return torch.clamp(perturbation, -epsilon, epsilon)
    
    def generate_adversarial_images(self, original_images, target_labels, epsilon):
        """
        Generate adversarial images from original images
        
        Args:
            original_images: Original images [B, 3, H, W]
            target_labels: Target class labels [B]
            epsilon: Maximum perturbation
        
        Returns:
            adversarial_images: Images with added perturbation
            perturbations: The perturbations themselves
        """
        # Generate perturbation
        perturbations = self.generator(original_images, target_labels)
        
        # Clamp perturbation
        perturbations = self.clamp_perturbation(perturbations, epsilon)
        
        # Add perturbation to original images
        adversarial_images = original_images + perturbations
        
        # Clamp to valid image range [0, 1] (assuming normalized images)
        # Note: If using ImageNet normalization, images are in different range
        # For normalized images, we don't clamp here, clamping happens during inference
        
        return adversarial_images, perturbations
    
    def train_discriminator(self, real_images, adversarial_images):
        """
        Train discriminator for one step
        
        Args:
            real_images: Real images [B, 3, H, W]
            adversarial_images: Adversarial images [B, 3, H, W]
        
        Returns:
            d_loss: Discriminator loss value
        """
        self.optimizer_D.zero_grad()
        
        # Real images -> label 1 (real)
        real_output = self.discriminator(real_images)
        d_loss_real = self.losses.adversarial_loss(real_output, is_real=True)
        
        # Adversarial images -> label 0 (fake)
        adv_output = self.discriminator(adversarial_images.detach())  # Detach to avoid backprop to G
        d_loss_fake = self.losses.adversarial_loss(adv_output, is_real=False)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        
        # Backward pass
        d_loss.backward()
        self.optimizer_D.step()
        
        return d_loss.item()
    
    def train_generator(self, original_images, target_labels):
        """
        Train generator for one step
        
        Args:
            original_images: Original images [B, 3, H, W]
            target_labels: Target class labels [B]
        
        Returns:
            g_loss: Total generator loss
            attack_success_rate: Percentage of successful attacks
        """
        self.optimizer_G.zero_grad()
        
        # Generate adversarial images
        adversarial_images, perturbations = self.generate_adversarial_images(
            original_images, target_labels, self.config['epsilon']
        )
        
        # 1. GAN adversarial loss
        # Generator wants discriminator to classify adversarial images as real
        adv_output = self.discriminator(adversarial_images)
        loss_gan = self.losses.adversarial_loss(adv_output, is_real=True)
        
        # 2. Attack loss
        # Generator wants surrogate model to classify adversarial images as target class
        loss_attack = self.losses.attack_loss(
            self.surrogate_model, adversarial_images, target_labels
        )
        
        # 3. Perturbation constraint loss
        loss_perturbation = self.losses.perturbation_loss(
            perturbations,
            norm=self.config['perturbation_norm'],
            epsilon=self.config['epsilon']
        )
        
        # 4. Feature-level loss (optional)
        if self.config['use_feature_loss']:
            loss_feature = self.losses.feature_loss(
                self.surrogate_model, original_images, adversarial_images
            )
        else:
            loss_feature = torch.tensor(0.0).to(self.device)
        
        # Total generator loss (weighted combination)
        g_loss = (
            self.config['lambda_adv'] * loss_gan +
            self.config['lambda_attack'] * loss_attack +
            self.config['lambda_perturbation'] * loss_perturbation +
            self.config['lambda_feature'] * loss_feature
        )
        
        # Backward pass
        g_loss.backward()
        self.optimizer_G.step()
        
        # Calculate attack success rate
        with torch.no_grad():
            outputs = self.surrogate_model(adversarial_images)
            _, predicted = outputs.max(1)
            attack_success_rate = (predicted == target_labels).float().mean().item() * 100
        
        return g_loss.item(), attack_success_rate
    
    def train_epoch(self, dataloader, epoch):
        """
        Train for one epoch
        
        Args:
            dataloader: Data loader (returns image, true_label, target_label)
            epoch: Current epoch number
        
        Returns:
            avg_g_loss, avg_d_loss, avg_attack_success_rate
        """
        self.generator.train()
        self.discriminator.train()
        
        g_losses = []
        d_losses = []
        attack_success_rates = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, true_labels, target_labels) in enumerate(pbar):
            images = images.to(self.device)
            target_labels = target_labels.to(self.device)
            
            # Generate adversarial images
            with torch.no_grad():
                adversarial_images, _ = self.generate_adversarial_images(
                    images, target_labels, self.config['epsilon']
                )
            
            # Train Discriminator n_critic times
            for _ in range(self.config['n_critic']):
                d_loss = self.train_discriminator(images, adversarial_images)
            
            # Train Generator once
            g_loss, attack_success = self.train_generator(images, target_labels)
            
            # Record losses
            g_losses.append(g_loss)
            d_losses.append(d_loss)
            attack_success_rates.append(attack_success)
            
            # Update progress bar
            if (batch_idx + 1) % self.config['log_interval'] == 0:
                pbar.set_postfix({
                    'G_loss': np.mean(g_losses[-10:]),
                    'D_loss': np.mean(d_losses[-10:]),
                    'Attack_SR': np.mean(attack_success_rates[-10:])
                })
        
        return np.mean(g_losses), np.mean(d_losses), np.mean(attack_success_rates)
    
    def train(self, dataloader, num_epochs, save_dir='checkpoints/gan/'):
        """
        Complete training loop
        
        Args:
            dataloader: Training data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        print("\n" + "="*80)
        print("GAN TRAINING FOR ADVERSARIAL ATTACK GENERATION")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Epsilon: {self.config['epsilon']}")
        print(f"Perturbation norm: {self.config['perturbation_norm']}")
        print(f"Generator LR: {self.config['lr_generator']}")
        print(f"Discriminator LR: {self.config['lr_discriminator']}")
        print("="*80 + "\n")
        
        os.makedirs(save_dir, exist_ok=True)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # Train for one epoch
            avg_g_loss, avg_d_loss, avg_attack_sr = self.train_epoch(dataloader, epoch)
            
            # Save to history
            self.history['g_loss'].append(avg_g_loss)
            self.history['d_loss'].append(avg_d_loss)
            self.history['attack_success_rate'].append(avg_attack_sr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs} Summary:")
            print(f"  Generator Loss: {avg_g_loss:.4f}")
            print(f"  Discriminator Loss: {avg_d_loss:.4f}")
            print(f"  Attack Success Rate: {avg_attack_sr:.2f}%")
            print("-" * 80)
            
            # Save checkpoint periodically
            if epoch % self.config['save_interval'] == 0:
                checkpoint_path = os.path.join(save_dir, f'gan_epoch_{epoch}.pth')
                self.save_checkpoint(checkpoint_path, epoch)
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        
        # Save final model
        final_path = os.path.join(save_dir, 'gan_final.pth')
        self.save_checkpoint(final_path, num_epochs)
        
        elapsed_time = time.time() - start_time
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print(f"Total Time: {elapsed_time/60:.2f} minutes")
        print(f"Final Attack Success Rate: {avg_attack_sr:.2f}%")
        print(f"Model saved to: {final_path}")
        print("="*80 + "\n")
    
    def save_checkpoint(self, path, epoch):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'history': self.history,
            'config': self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch']


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================
def main():
    """
    Main function to train the GAN
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GAN for adversarial attacks")
    parser.add_argument(
        '--surrogate',
        type=str,
        required=True,
        choices=['vgg16', 'resnet50', 'efficientnet'],
        help="Which surrogate model to use"
    )
    parser.add_argument(
        '--surrogate_checkpoint',
        type=str,
        required=True,
        help="Path to surrogate model checkpoint"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help="Path to training dataset"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    print("Loading surrogate model...")
    surrogate_model = load_surrogate_model(
        model_type=args.surrogate,
        checkpoint_path=args.surrogate_checkpoint,
        num_classes=config.NUM_CLASSES,
        device=config.DEVICE
    )
    print("✓ Surrogate model loaded")
    
    print("\nLoading dataset...")
    dataloader = load_gan_dataset(
        dataset_path=args.dataset,
        batch_size=config.GAN_CONFIG['batch_size'],
        num_workers=4
    )
    print("✓ Dataset loaded")
    
    print("\nInitializing GAN...")
    generator = Generator(
        num_classes=config.NUM_CLASSES,
        latent_dim=config.GAN_CONFIG['latent_dim'],
        channels=config.GAN_CONFIG['generator_channels']
    )
    generator.apply(weights_init)
    
    discriminator = Discriminator(
        channels=config.GAN_CONFIG['discriminator_channels']
    )
    discriminator.apply(weights_init)
    print("✓ GAN initialized")
    
    # Create trainer
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        surrogate_model=surrogate_model,
        device=config.DEVICE,
        config_dict=config.GAN_CONFIG
    )
    
    # Train
    trainer.train(
        dataloader=dataloader,
        num_epochs=config.GAN_CONFIG['num_epochs'],
        save_dir='checkpoints/gan/'
    )


if __name__ == "__main__":
    main()