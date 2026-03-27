"""
Configuration file for adversarial machine learning framework
Contains all hyperparameters and settings for surrogate models and GAN training
"""

import torch

# ============================================================
# GENERAL SETTINGS
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 8  # Number of malware classes
IMAGE_SIZE = 224  # Standard input size for ResNet/VGG/etc.
RANDOM_SEED = 42

# ============================================================
# DATASET PATHS
# ============================================================
# Model B's dataset (your custom malware dataset)
DATASET_B_TRAIN = "/home/beylessen/Desktop/PFA2/Sorted/train"
DATASET_B_VAL = "/home/beylessen/Desktop/PFA2/Sorted/val"
DATASET_B_TEST = "/home/beylessen/Desktop/PFA2/Sorted/test"

# Cross-domain dataset (e.g., Malimg or another malware dataset)
DATASET_CROSS_DOMAIN_TRAIN = "path/to/malimg/train"
DATASET_CROSS_DOMAIN_VAL = "path/to/malimg/val"
DATASET_CROSS_DOMAIN_TEST = "path/to/malimg/test"

# ============================================================
# MODEL CHECKPOINTS
# ============================================================
MODEL_B_CHECKPOINT = "checkpoints/model_b_resnet50.pth"
MODEL_A_CHECKPOINT = "checkpoints/model_a_vgg16.pth"
MODEL_A_PRIME_CHECKPOINT = "checkpoints/model_a_prime_resnet50.pth"
MODEL_A_DOUBLE_PRIME_CHECKPOINT = "checkpoints/model_a_double_prime_efficientnet.pth"

# ============================================================
# SURROGATE MODEL TRAINING
# ============================================================
SURROGATE_CONFIG = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'scheduler_step_size': 15,
    'scheduler_gamma': 0.1,
    'early_stopping_patience': 10,
    'num_workers': 4,
}

# ============================================================
# GAN CONFIGURATION
# ============================================================
GAN_CONFIG = {
    # Architecture
    'latent_dim': 100,  # Size of target class embedding
    'generator_channels': [64, 128, 256, 512],  # Encoder-decoder channels
    'discriminator_channels': [64, 128, 256, 512],
    
    # Training hyperparameters
    'batch_size': 16,
    'num_epochs': 100,
    'lr_generator': 0.0002,
    'lr_discriminator': 0.0002,
    'beta1': 0.5,  # Adam optimizer beta1
    'beta2': 0.999,  # Adam optimizer beta2
    
    # Loss weights (CRITICAL - tune these for your task)
    'lambda_adv': 1.0,      # GAN adversarial loss weight
    'lambda_attack': 10.0,   # Attack loss (cross-entropy) weight
    'lambda_perturbation': 0.5,  # Perturbation constraint weight
    'lambda_feature': 1.0,   # Feature-level loss weight (optional)
    
    # Perturbation constraints
    'epsilon': 0.1,  # Maximum perturbation magnitude (L-infinity)
    'perturbation_norm': 'linf',  # 'l2' or 'linf'
    
    # Training settings
    'n_critic': 5,  # Train discriminator n_critic times per generator update
    'gradient_penalty_weight': 10.0,  # For WGAN-GP (optional)
    'use_feature_loss': True,  # Whether to use feature-level loss
    
    # Logging
    'log_interval': 50,  # Log every N batches
    'save_interval': 5,  # Save checkpoint every N epochs
}

# ============================================================
# DATA AUGMENTATION
# ============================================================
# For surrogate model training
TRAIN_AUGMENTATION = True
AUGMENTATION_CONFIG = {
    'horizontal_flip': 0.5,
    'rotation_degrees': 15,
    'color_jitter': {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1,
    }
}

# ============================================================
# NORMALIZATION (ImageNet statistics - adjust if needed)
# ============================================================
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ============================================================
# ADVERSARIAL EXAMPLE GENERATION
# ============================================================
ATTACK_CONFIG = {
    'target_class': None,  # Set during runtime, or None for untargeted
    'epsilon': 0.1,  # Same as training epsilon
    'save_adversarial_images': True,
    'output_dir': 'adversarial_examples/',
}