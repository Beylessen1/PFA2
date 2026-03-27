"""
Adversarial Transferability Evaluation Script
==============================================
Evaluates how well adversarial examples generated on Model A transfer to Model B.

Usage:
    python evaluate_transferability.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt


def evaluate_transferability(
    model: torch.nn.Module,
    adv_images: torch.Tensor,
    true_labels: torch.Tensor,
    target_labels: Optional[torch.Tensor] = None,
    device: str = 'cuda',
    batch_size: int = 64,
    return_predictions: bool = False
) -> Dict[str, float]:
    """
    Evaluate adversarial transferability on a target model.
    
    Args:
        model: Target model (Model B)
        adv_images: Adversarial images tensor (N, C, H, W)
        true_labels: True class labels (N,)
        target_labels: Target labels for targeted attacks (N,) [optional]
        device: 'cuda' or 'cpu'
        batch_size: Batch size for evaluation
        return_predictions: If True, also return predictions and confidences
        
    Returns:
        Dictionary containing:
            - transferability_rate: % of samples misclassified
            - targeted_success_rate: % of samples classified as target (if target_labels provided)
            - avg_confidence: Average confidence of model predictions
            - avg_success_confidence: Average confidence for successful attacks
            - total_samples: Total number of samples evaluated
            - predictions: Model predictions (if return_predictions=True)
            - confidences: Prediction confidences (if return_predictions=True)
    """
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    # Create DataLoader
    if target_labels is not None:
        dataset = TensorDataset(adv_images, true_labels, target_labels)
    else:
        dataset = TensorDataset(adv_images, true_labels)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize tracking variables
    total_samples = 0
    misclassified = 0  # For untargeted success
    targeted_success = 0  # For targeted success
    
    all_predictions = []
    all_confidences = []
    all_success_confidences = []
    
    # Evaluate
    with torch.no_grad():
        for batch_data in dataloader:
            if target_labels is not None:
                batch_adv, batch_true, batch_target = batch_data
                batch_target = batch_target.to(device)
            else:
                batch_adv, batch_true = batch_data
                batch_target = None
            
            batch_adv = batch_adv.to(device)
            batch_true = batch_true.to(device)
            
            # Forward pass
            outputs = model(batch_adv)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get predictions and confidences
            confidences, predictions = torch.max(probabilities, dim=1)
            
            # Calculate metrics
            total_samples += batch_true.size(0)
            
            # Untargeted success: prediction != true label
            misclassified += (predictions != batch_true).sum().item()
            
            # Targeted success: prediction == target label
            if batch_target is not None:
                targeted_success += (predictions == batch_target).sum().item()
                
                # Track confidences for successful targeted attacks
                success_mask = (predictions == batch_target)
                if success_mask.any():
                    all_success_confidences.extend(
                        confidences[success_mask].cpu().numpy().tolist()
                    )
            else:
                # For untargeted attacks, successful = misclassified
                success_mask = (predictions != batch_true)
                if success_mask.any():
                    all_success_confidences.extend(
                        confidences[success_mask].cpu().numpy().tolist()
                    )
            
            # Store predictions and confidences
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_confidences.extend(confidences.cpu().numpy().tolist())
    
    # Calculate rates
    transferability_rate = 100.0 * misclassified / total_samples
    targeted_success_rate = 100.0 * targeted_success / total_samples if target_labels is not None else None
    avg_confidence = np.mean(all_confidences)
    avg_success_confidence = np.mean(all_success_confidences) if all_success_confidences else 0.0
    
    # Prepare results
    results = {
        'transferability_rate': transferability_rate,
        'targeted_success_rate': targeted_success_rate,
        'avg_confidence': avg_confidence,
        'avg_success_confidence': avg_success_confidence,
        'total_samples': total_samples,
        'misclassified': misclassified,
        'targeted_success': targeted_success if target_labels is not None else None,
    }
    
    if return_predictions:
        results['predictions'] = np.array(all_predictions)
        results['confidences'] = np.array(all_confidences)
    
    return results


def evaluate_transferability_from_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    has_target_labels: bool = False
) -> Dict[str, float]:
    """
    Evaluate adversarial transferability using a DataLoader.
    
    Args:
        model: Target model (Model B)
        dataloader: DataLoader yielding (adv_images, true_labels) or 
                    (adv_images, true_labels, target_labels)
        device: 'cuda' or 'cpu'
        has_target_labels: Whether the dataloader includes target labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    
    model = model.to(device)
    model.eval()
    
    total_samples = 0
    misclassified = 0
    targeted_success = 0
    
    all_confidences = []
    all_success_confidences = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            if has_target_labels:
                batch_adv, batch_true, batch_target = batch_data
                batch_target = batch_target.to(device)
            else:
                batch_adv, batch_true = batch_data
                batch_target = None
            
            batch_adv = batch_adv.to(device)
            batch_true = batch_true.to(device)
            
            # Forward pass
            outputs = model(batch_adv)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)
            
            # Calculate metrics
            total_samples += batch_true.size(0)
            misclassified += (predictions != batch_true).sum().item()
            
            if batch_target is not None:
                targeted_success += (predictions == batch_target).sum().item()
                success_mask = (predictions == batch_target)
            else:
                success_mask = (predictions != batch_true)
            
            # Track confidences
            all_confidences.extend(confidences.cpu().numpy().tolist())
            if success_mask.any():
                all_success_confidences.extend(
                    confidences[success_mask].cpu().numpy().tolist()
                )
    
    # Calculate rates
    results = {
        'transferability_rate': 100.0 * misclassified / total_samples,
        'targeted_success_rate': 100.0 * targeted_success / total_samples if has_target_labels else None,
        'avg_confidence': np.mean(all_confidences),
        'avg_success_confidence': np.mean(all_success_confidences) if all_success_confidences else 0.0,
        'total_samples': total_samples,
        'misclassified': misclassified,
        'targeted_success': targeted_success if has_target_labels else None,
    }
    
    return results


def print_evaluation_results(results: Dict[str, float], attack_type: str = "untargeted"):
    """
    Pretty print evaluation results.
    
    Args:
        results: Results dictionary from evaluate_transferability
        attack_type: "targeted" or "untargeted"
    """
    
    print("\n" + "="*60)
    print("ADVERSARIAL TRANSFERABILITY EVALUATION RESULTS")
    print("="*60)
    print(f"Attack Type: {attack_type.upper()}")
    print(f"Total Samples: {results['total_samples']}")
    print("-"*60)
    
    # Untargeted metrics
    print(f"Transferability Rate: {results['transferability_rate']:.2f}%")
    print(f"  → Misclassified: {results['misclassified']}/{results['total_samples']}")
    
    # Targeted metrics
    if results['targeted_success_rate'] is not None:
        print(f"\nTargeted Success Rate: {results['targeted_success_rate']:.2f}%")
        print(f"  → Correctly Targeted: {results['targeted_success']}/{results['total_samples']}")
    
    # Confidence metrics
    print(f"\nAverage Confidence: {results['avg_confidence']:.4f}")
    print(f"Avg Confidence (Successful Attacks): {results['avg_success_confidence']:.4f}")
    
    print("="*60 + "\n")


def visualize_adversarial_examples(
    adv_images: torch.Tensor,
    true_labels: torch.Tensor,
    predictions: np.ndarray,
    confidences: np.ndarray,
    target_labels: Optional[torch.Tensor] = None,
    class_names: Optional[list] = None,
    num_samples: int = 8,
    denormalize: bool = True,
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225]
):
    """
    Visualize adversarial examples with predictions.
    
    Args:
        adv_images: Adversarial images (N, C, H, W)
        true_labels: True labels (N,)
        predictions: Model predictions (N,)
        confidences: Prediction confidences (N,)
        target_labels: Target labels for targeted attacks (N,) [optional]
        class_names: List of class names [optional]
        num_samples: Number of samples to visualize
        denormalize: Whether to denormalize images
        mean: Mean values for denormalization
        std: Std values for denormalization
    """
    
    # Color scheme from your training code
    pink = "#C11C84"
    node_black = "#141D2B"
    hacker_grey = "#A4B1CD"
    
    # Select random samples
    num_samples = min(num_samples, len(adv_images))
    indices = np.random.choice(len(adv_images), num_samples, replace=False)
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor=node_black)
    axes = axes.flatten()
    
    for idx, img_idx in enumerate(indices):
        img = adv_images[img_idx].cpu().clone()
        
        # Denormalize if needed
        if denormalize:
            for c in range(3):
                img[c] = img[c] * std[c] + mean[c]
        
        # Convert to numpy and transpose
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        # Get labels
        true_label = true_labels[img_idx].item()
        pred_label = predictions[img_idx]
        conf = confidences[img_idx]
        
        # Determine if attack was successful
        is_successful = (pred_label != true_label)
        
        # Create title
        if class_names:
            true_name = class_names[true_label]
            pred_name = class_names[pred_label]
            title = f"True: {true_name}\nPred: {pred_name}\nConf: {conf:.2%}"
            
            if target_labels is not None:
                target_label = target_labels[img_idx].item()
                target_name = class_names[target_label]
                title = f"True: {true_name}\nTarget: {target_name}\nPred: {pred_name}\nConf: {conf:.2%}"
        else:
            title = f"True: {true_label}\nPred: {pred_label}\nConf: {conf:.2%}"
            
            if target_labels is not None:
                target_label = target_labels[img_idx].item()
                title = f"True: {true_label}\nTarget: {target_label}\nPred: {pred_label}\nConf: {conf:.2%}"
        
        # Plot
        axes[idx].imshow(img)
        axes[idx].set_title(
            title, 
            color=pink if is_successful else hacker_grey,
            fontsize=10
        )
        axes[idx].axis('off')
    
    # Style the figure
    plt.suptitle(
        "Adversarial Examples (Pink = Successful Transfer)", 
        color=pink, 
        fontsize=16,
        y=0.98
    )
    plt.tight_layout()
    plt.show()


def create_confusion_summary(
    true_labels: torch.Tensor,
    predictions: np.ndarray,
    class_names: Optional[list] = None
) -> Dict[int, Dict[int, int]]:
    """
    Create a confusion summary showing how true labels were misclassified.
    
    Args:
        true_labels: True labels (N,)
        predictions: Model predictions (N,)
        class_names: List of class names [optional]
        
    Returns:
        Dictionary mapping true_class -> {predicted_class: count}
    """
    
    true_labels = true_labels.cpu().numpy()
    
    confusion = {}
    for true_label in np.unique(true_labels):
        confusion[int(true_label)] = {}
        mask = (true_labels == true_label)
        preds_for_class = predictions[mask]
        
        for pred_label in np.unique(preds_for_class):
            count = np.sum(preds_for_class == pred_label)
            confusion[int(true_label)][int(pred_label)] = int(count)
    
    # Print summary
    print("\nCONFUSION SUMMARY")
    print("="*60)
    for true_label, pred_dict in confusion.items():
        true_name = class_names[true_label] if class_names else str(true_label)
        print(f"\nTrue Class: {true_name}")
        for pred_label, count in sorted(pred_dict.items(), key=lambda x: x[1], reverse=True):
            pred_name = class_names[pred_label] if class_names else str(pred_label)
            print(f"  → Predicted as {pred_name}: {count}")
    print("="*60)
    
    return confusion


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage demonstrating how to evaluate adversarial transferability.
    """
    
    import torch.nn as nn
    import torchvision.models as models
    
    # ========================================================================
    # 1. DEFINE YOUR MODEL B (same architecture as in your training code)
    # ========================================================================
    
    class MalwareClassifier(nn.Module):
        def __init__(self, n_classes):
            super(MalwareClassifier, self).__init__()
            self.resnet = models.resnet50(weights='DEFAULT')
            
            for param in self.resnet.parameters():
                param.requires_grad = False
            
            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Sequential(
                nn.Linear(num_features, 1000),
                nn.ReLU(),
                nn.Linear(1000, n_classes)
            )
        
        def forward(self, x):
            return self.resnet(x)
    
    # ========================================================================
    # 2. LOAD YOUR TRAINED MODEL B
    # ========================================================================
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    n_classes = 8  # Your malware classes
    model_B = MalwareClassifier(n_classes)
    
    # Load trained weights
    # Option 1: If you saved with torch.save(model.state_dict(), ...)
    # model_B.load_state_dict(torch.load("malware_classifier.pth"))
    
    # Option 2: If you saved with torch.jit.script (as in your code)
    # model_B = torch.jit.load("malware_classifier.pth")
    
    model_B = model_B.to(device)
    model_B.eval()
    
    # ========================================================================
    # 3. LOAD YOUR ADVERSARIAL EXAMPLES
    # ========================================================================
    
    # Example: Load from saved tensors
    # adv_images = torch.load("adversarial_images.pt")
    # true_labels = torch.load("true_labels.pt")
    # target_labels = torch.load("target_labels.pt")  # Optional, for targeted attacks
    
    # For demonstration, create dummy data
    print("\n[DEMO] Creating dummy adversarial data...")
    batch_size = 100
    adv_images = torch.randn(batch_size, 3, 224, 224)  # Replace with your actual data
    true_labels = torch.randint(0, n_classes, (batch_size,))
    target_labels = torch.randint(0, n_classes, (batch_size,))  # For targeted attacks
    
    # ========================================================================
    # 4. EVALUATE TRANSFERABILITY
    # ========================================================================
    
    print("\n[i] Evaluating adversarial transferability...")
    
    # For TARGETED attacks
    results_targeted = evaluate_transferability(
        model=model_B,
        adv_images=adv_images,
        true_labels=true_labels,
        target_labels=target_labels,
        device=device,
        batch_size=32,
        return_predictions=True
    )
    
    print_evaluation_results(results_targeted, attack_type="targeted")
    
    # For UNTARGETED attacks
    results_untargeted = evaluate_transferability(
        model=model_B,
        adv_images=adv_images,
        true_labels=true_labels,
        target_labels=None,
        device=device,
        batch_size=32,
        return_predictions=True
    )
    
    print_evaluation_results(results_untargeted, attack_type="untargeted")
    
    # ========================================================================
    # 5. VISUALIZE EXAMPLES (OPTIONAL)
    # ========================================================================
    
    class_names = [
        "adware", "backdoor", "benign", "downloader",
        "spyware", "trojan", "virus", "worm"
    ]
    
    print("\n[i] Visualizing adversarial examples...")
    
    visualize_adversarial_examples(
        adv_images=adv_images,
        true_labels=true_labels,
        predictions=results_targeted['predictions'],
        confidences=results_targeted['confidences'],
        target_labels=target_labels,
        class_names=class_names,
        num_samples=8,
        denormalize=True
    )
    
    # ========================================================================
    # 6. CONFUSION SUMMARY (OPTIONAL)
    # ========================================================================
    
    create_confusion_summary(
        true_labels=true_labels,
        predictions=results_targeted['predictions'],
        class_names=class_names
    )