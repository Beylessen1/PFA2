"""
Adversarial Example Generation
Use trained GAN to generate adversarial examples for testing transferability
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

import config
from gan_models import Generator
from surrogate_models import load_surrogate_model
from data_utils import get_test_transforms, denormalize, save_image


# ============================================================
# ADVERSARIAL EXAMPLE GENERATOR
# ============================================================
class AdversarialGenerator:
    """
    Generate adversarial examples using trained GAN
    """
    
    def __init__(self, generator_checkpoint, device='cuda'):
        """
        Args:
            generator_checkpoint: Path to trained generator checkpoint
            device: Device to run on
        """
        self.device = device
        
        # Load generator
        print("Loading trained generator...")
        checkpoint = torch.load(generator_checkpoint, map_location=device)
        
        self.generator = Generator(
            num_classes=config.NUM_CLASSES,
            latent_dim=checkpoint['config']['latent_dim'],
            channels=checkpoint['config']['generator_channels']
        )
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator = self.generator.to(device)
        self.generator.eval()
        
        self.epsilon = checkpoint['config']['epsilon']
        
        print(f"✓ Generator loaded (epsilon={self.epsilon})")
    
    def generate(self, image, target_class):
        """
        Generate adversarial example for a single image
        
        Args:
            image: Input image tensor [3, H, W] or [1, 3, H, W] (normalized)
            target_class: Target class (integer)
        
        Returns:
            adversarial_image: Adversarial image [3, H, W] (normalized)
            perturbation: Perturbation [3, H, W]
        """
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)  # [1, 3, H, W]
        
        image = image.to(self.device)
        target_label = torch.tensor([target_class], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # Generate perturbation
            perturbation = self.generator(image, target_label)
            
            # Clamp perturbation
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
            
            # Add perturbation to image
            adversarial_image = image + perturbation
            
            # Note: We don't clamp to [0, 1] here because images are normalized
            # The clamping will happen during evaluation or when saving
        
        return adversarial_image.squeeze(0), perturbation.squeeze(0)
    
    def generate_batch(self, images, target_classes):
        """
        Generate adversarial examples for a batch of images
        
        Args:
            images: Batch of images [B, 3, H, W] (normalized)
            target_classes: Target classes [B] (integers or list)
        
        Returns:
            adversarial_images: Batch of adversarial images [B, 3, H, W]
            perturbations: Batch of perturbations [B, 3, H, W]
        """
        images = images.to(self.device)
        
        if isinstance(target_classes, list):
            target_classes = torch.tensor(target_classes, dtype=torch.long)
        target_classes = target_classes.to(self.device)
        
        with torch.no_grad():
            # Generate perturbations
            perturbations = self.generator(images, target_classes)
            
            # Clamp perturbations
            perturbations = torch.clamp(perturbations, -self.epsilon, self.epsilon)
            
            # Add perturbations to images
            adversarial_images = images + perturbations
        
        return adversarial_images, perturbations


# ============================================================
# TRANSFERABILITY EVALUATION
# ============================================================
class TransferabilityEvaluator:
    """
    Evaluate transferability of adversarial examples across models
    """
    
    def __init__(self, target_model, device='cuda'):
        """
        Args:
            target_model: The target model (Model B) to test transferability on
            device: Device to run on
        """
        self.target_model = target_model.to(device)
        self.target_model.eval()
        self.device = device
        
        # Freeze target model
        for param in self.target_model.parameters():
            param.requires_grad = False
    
    def evaluate_single(self, original_image, adversarial_image, true_label, target_label):
        """
        Evaluate a single adversarial example
        
        Args:
            original_image: Original image [3, H, W] or [1, 3, H, W]
            adversarial_image: Adversarial image [3, H, W] or [1, 3, H, W]
            true_label: True class (integer)
            target_label: Target class (integer)
        
        Returns:
            dict with evaluation metrics
        """
        # Ensure batch dimension
        if original_image.dim() == 3:
            original_image = original_image.unsqueeze(0)
        if adversarial_image.dim() == 3:
            adversarial_image = adversarial_image.unsqueeze(0)
        
        original_image = original_image.to(self.device)
        adversarial_image = adversarial_image.to(self.device)
        
        with torch.no_grad():
            # Predict on original image
            orig_output = self.target_model(original_image)
            orig_pred = orig_output.argmax(dim=1).item()
            
            # Predict on adversarial image
            adv_output = self.target_model(adversarial_image)
            adv_pred = adv_output.argmax(dim=1).item()
            
            # Get confidence scores
            orig_confidence = F.softmax(orig_output, dim=1)[0, orig_pred].item()
            adv_confidence = F.softmax(adv_output, dim=1)[0, adv_pred].item()
        
        # Calculate metrics
        results = {
            'original_prediction': orig_pred,
            'adversarial_prediction': adv_pred,
            'true_label': true_label,
            'target_label': target_label,
            'original_confidence': orig_confidence,
            'adversarial_confidence': adv_confidence,
            'transfer_success': adv_pred == target_label,  # Targeted attack success
            'misclassification_success': adv_pred != true_label,  # Untargeted attack success
        }
        
        return results
    
    def evaluate_batch(self, original_images, adversarial_images, true_labels, target_labels):
        """
        Evaluate a batch of adversarial examples
        
        Args:
            original_images: Batch of original images [B, 3, H, W]
            adversarial_images: Batch of adversarial images [B, 3, H, W]
            true_labels: True class labels [B]
            target_labels: Target class labels [B]
        
        Returns:
            dict with aggregated metrics
        """
        original_images = original_images.to(self.device)
        adversarial_images = adversarial_images.to(self.device)
        true_labels = true_labels.to(self.device)
        target_labels = target_labels.to(self.device)
        
        batch_size = original_images.size(0)
        
        with torch.no_grad():
            # Predictions on original images
            orig_outputs = self.target_model(original_images)
            orig_preds = orig_outputs.argmax(dim=1)
            
            # Predictions on adversarial images
            adv_outputs = self.target_model(adversarial_images)
            adv_preds = adv_outputs.argmax(dim=1)
        
        # Calculate metrics
        transfer_success = (adv_preds == target_labels).float().sum().item()
        misclassification_success = (adv_preds != true_labels).float().sum().item()
        original_accuracy = (orig_preds == true_labels).float().sum().item()
        
        results = {
            'batch_size': batch_size,
            'transfer_success_rate': (transfer_success / batch_size) * 100,
            'misclassification_rate': (misclassification_success / batch_size) * 100,
            'original_accuracy': (original_accuracy / batch_size) * 100,
        }
        
        return results


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def load_single_image(image_path, transform=None):
    """
    Load and preprocess a single image
    
    Args:
        image_path: Path to image
        transform: Torchvision transform to apply
    
    Returns:
        Preprocessed image tensor
    """
    if transform is None:
        transform = get_test_transforms()
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    
    return image_tensor


def save_adversarial_example(original_image, adversarial_image, perturbation, save_dir, filename):
    """
    Save original, adversarial, and perturbation images
    
    Args:
        original_image: Original image tensor [3, H, W] (normalized)
        adversarial_image: Adversarial image tensor [3, H, W] (normalized)
        perturbation: Perturbation tensor [3, H, W]
        save_dir: Directory to save images
        filename: Base filename (without extension)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Denormalize images for saving
    orig_denorm = denormalize(original_image.cpu())
    adv_denorm = denormalize(adversarial_image.cpu())
    
    # Normalize perturbation for visualization (map to [0, 1])
    pert_vis = (perturbation.cpu() + 1) / 2  # Map from [-1, 1] to [0, 1]
    
    # Save images
    save_image(orig_denorm, os.path.join(save_dir, f"{filename}_original.png"))
    save_image(adv_denorm, os.path.join(save_dir, f"{filename}_adversarial.png"))
    save_image(pert_vis, os.path.join(save_dir, f"{filename}_perturbation.png"))


def generate_and_evaluate_dataset(
    generator_checkpoint,
    target_model_checkpoint,
    target_model_type,
    dataset_loader,
    save_examples=True,
    output_dir='adversarial_examples/'
):
    """
    Generate adversarial examples for entire dataset and evaluate transferability
    
    Args:
        generator_checkpoint: Path to trained generator checkpoint
        target_model_checkpoint: Path to target model (Model B) checkpoint
        target_model_type: Type of target model ('resnet50', etc.)
        dataset_loader: DataLoader for the dataset
        save_examples: Whether to save example images
        output_dir: Directory to save examples
    
    Returns:
        Overall evaluation metrics
    """
    print("\n" + "="*80)
    print("GENERATING ADVERSARIAL EXAMPLES AND EVALUATING TRANSFERABILITY")
    print("="*80)
    
    # Initialize generator
    adv_gen = AdversarialGenerator(generator_checkpoint, device=config.DEVICE)
    
    # Load target model
    print(f"\nLoading target model ({target_model_type})...")
    target_model = load_surrogate_model(
        model_type=target_model_type,
        checkpoint_path=target_model_checkpoint,
        num_classes=config.NUM_CLASSES,
        device=config.DEVICE
    )
    print("✓ Target model loaded")
    
    # Initialize evaluator
    evaluator = TransferabilityEvaluator(target_model, device=config.DEVICE)
    
    # Evaluation metrics
    total_samples = 0
    total_transfer_success = 0
    total_misclassification = 0
    total_original_correct = 0
    
    print("\nGenerating adversarial examples...")
    
    for batch_idx, (images, true_labels, target_labels) in enumerate(tqdm(dataset_loader)):
        images = images.to(config.DEVICE)
        
        # Generate adversarial examples
        adversarial_images, perturbations = adv_gen.generate_batch(images, target_labels)
        
        # Evaluate
        results = evaluator.evaluate_batch(images, adversarial_images, true_labels, target_labels)
        
        # Accumulate metrics
        total_samples += results['batch_size']
        total_transfer_success += results['transfer_success_rate'] * results['batch_size'] / 100
        total_misclassification += results['misclassification_rate'] * results['batch_size'] / 100
        total_original_correct += results['original_accuracy'] * results['batch_size'] / 100
        
        # Save some example images
        if save_examples and batch_idx < 5:  # Save first 5 batches
            for i in range(min(4, images.size(0))):  # Save up to 4 images per batch
                filename = f"batch{batch_idx}_sample{i}"
                save_adversarial_example(
                    images[i],
                    adversarial_images[i],
                    perturbations[i],
                    output_dir,
                    filename
                )
    
    # Calculate overall metrics
    overall_metrics = {
        'total_samples': total_samples,
        'transfer_success_rate': (total_transfer_success / total_samples) * 100,
        'misclassification_rate': (total_misclassification / total_samples) * 100,
        'original_accuracy': (total_original_correct / total_samples) * 100,
    }
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Total Samples: {overall_metrics['total_samples']}")
    print(f"Original Model Accuracy: {overall_metrics['original_accuracy']:.2f}%")
    print(f"Targeted Transfer Success Rate: {overall_metrics['transfer_success_rate']:.2f}%")
    print(f"Untargeted Misclassification Rate: {overall_metrics['misclassification_rate']:.2f}%")
    print("="*80 + "\n")
    
    return overall_metrics


# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    """
    Main function for generating and evaluating adversarial examples
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate and evaluate adversarial examples")
    parser.add_argument(
        '--generator_checkpoint',
        type=str,
        required=True,
        help="Path to trained generator checkpoint"
    )
    parser.add_argument(
        '--target_model_checkpoint',
        type=str,
        default=config.MODEL_B_CHECKPOINT,
        help="Path to target model checkpoint (Model B)"
    )
    parser.add_argument(
        '--target_model_type',
        type=str,
        default='resnet50',
        help="Type of target model"
    )
    parser.add_argument(
        '--image_path',
        type=str,
        help="Path to single image (for single image mode)"
    )
    parser.add_argument(
        '--target_class',
        type=int,
        help="Target class for attack"
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        help="Path to dataset (for batch mode)"
    )
    
    args = parser.parse_args()
    
    if args.image_path:
        # Single image mode
        print("Single image mode")
        
        # Load image
        image = load_single_image(args.image_path)
        
        # Initialize generator
        adv_gen = AdversarialGenerator(args.generator_checkpoint, device=config.DEVICE)
        
        # Generate adversarial example
        adversarial_image, perturbation = adv_gen.generate(image, args.target_class)
        
        # Save results
        save_adversarial_example(
            image, adversarial_image, perturbation,
            config.ATTACK_CONFIG['output_dir'],
            'single_example'
        )
        
        print(f"\nAdversarial example saved to {config.ATTACK_CONFIG['output_dir']}")
        
        # Evaluate if target model provided
        if args.target_model_checkpoint:
            print("\nEvaluating on target model...")
            target_model = load_surrogate_model(
                model_type=args.target_model_type,
                checkpoint_path=args.target_model_checkpoint,
                num_classes=config.NUM_CLASSES,
                device=config.DEVICE
            )
            
            evaluator = TransferabilityEvaluator(target_model, device=config.DEVICE)
            results = evaluator.evaluate_single(
                image, adversarial_image, None, args.target_class
            )
            
            print("\nResults:")
            print(f"  Adversarial prediction: {results['adversarial_prediction']}")
            print(f"  Target class: {results['target_label']}")
            print(f"  Transfer success: {results['transfer_success']}")
            print(f"  Adversarial confidence: {results['adversarial_confidence']:.4f}")
    
    elif args.dataset_path:
        # Batch mode
        print("Batch mode - evaluating entire dataset")
        
        from data_utils import load_gan_dataset
        
        dataloader = load_gan_dataset(
            dataset_path=args.dataset_path,
            batch_size=32,
            num_workers=4
        )
        
        metrics = generate_and_evaluate_dataset(
            generator_checkpoint=args.generator_checkpoint,
            target_model_checkpoint=args.target_model_checkpoint,
            target_model_type=args.target_model_type,
            dataset_loader=dataloader,
            save_examples=config.ATTACK_CONFIG['save_adversarial_images'],
            output_dir=config.ATTACK_CONFIG['output_dir']
        )
    
    else:
        print("Error: Must provide either --image_path or --dataset_path")


if __name__ == "__main__":
    main()