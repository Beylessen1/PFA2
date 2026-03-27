"""
Surrogate Model Architectures
Defines the three surrogate models for transferability experiments:
- Model A: Cross-architecture (VGG16)
- Model A': Cross-domain (ResNet-50)
- Model A'': Cross-architecture + Cross-domain (EfficientNet-B0)
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ============================================================
# MODEL A: Cross-Architecture (VGG16)
# Same dataset as Model B, different architecture
# ============================================================
class ModelA_VGG16(nn.Module):
    """
    VGG16-based surrogate model
    Trained on the SAME dataset as Model B (target)
    Different architecture from ResNet-50
    """
    def __init__(self, num_classes=8, pretrained=True):
        super(ModelA_VGG16, self).__init__()
        
        # Load pretrained VGG16
        self.model = models.vgg16(pretrained=pretrained)
        
        # Replace the final classifier layer for num_classes
        # VGG16 classifier: (classifier): Sequential with 7 layers
        # Last layer is Linear(4096, 1000) - we replace this
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def get_features(self, x):
        """
        Extract intermediate features for feature-level loss
        Returns features from the last convolutional layer
        """
        # VGG16 features are in self.model.features
        features = self.model.features(x)
        features = self.model.avgpool(features)
        features = torch.flatten(features, 1)
        return features


# ============================================================
# MODEL A': Cross-Domain (ResNet-50)
# Same architecture as Model B, different dataset
# ============================================================
class ModelAPrime_ResNet50(nn.Module):
    """
    ResNet-50 based surrogate model
    SAME architecture as Model B (ResNet-50)
    Trained on a DIFFERENT dataset (e.g., Malimg)
    """
    def __init__(self, num_classes=8, pretrained=True):
        super(ModelAPrime_ResNet50, self).__init__()
        
        # Load pretrained ResNet-50
        self.model = models.resnet50(pretrained=pretrained)
        
        # Replace the final fully connected layer
        # ResNet-50: fc is Linear(2048, 1000) by default
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def get_features(self, x):
        """
        Extract intermediate features for feature-level loss
        Returns features before the final FC layer
        """
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)
        return features


# ============================================================
# MODEL A'': Cross-Architecture + Cross-Domain (EfficientNet-B0)
# Different architecture AND different dataset
# ============================================================
class ModelADoublePrime_EfficientNet(nn.Module):
    """
    EfficientNet-B0 based surrogate model
    Different architecture from ResNet-50
    Trained on a DIFFERENT dataset (same as A')
    """
    def __init__(self, num_classes=8, pretrained=True):
        super(ModelADoublePrime_EfficientNet, self).__init__()
        
        # Load pretrained EfficientNet-B0
        if pretrained:
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        else:
            self.model = models.efficientnet_b0(weights=None)
        
        # Replace the final classifier layer
        # EfficientNet: classifier[1] is Linear(1280, 1000)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def get_features(self, x):
        """
        Extract intermediate features for feature-level loss
        Returns features before the final classifier
        """
        # EfficientNet features
        features = self.model.features(x)
        features = self.model.avgpool(features)
        features = torch.flatten(features, 1)
        return features


# ============================================================
# ALTERNATIVE: DenseNet-121 (another cross-architecture option)
# ============================================================
class ModelA_DenseNet121(nn.Module):
    """
    DenseNet-121 based surrogate model
    Alternative to VGG16 for cross-architecture experiments
    """
    def __init__(self, num_classes=8, pretrained=True):
        super(ModelA_DenseNet121, self).__init__()
        
        # Load pretrained DenseNet-121
        self.model = models.densenet121(pretrained=pretrained)
        
        # Replace the final classifier layer
        # DenseNet: classifier is Linear(1024, 1000)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def get_features(self, x):
        """
        Extract intermediate features for feature-level loss
        """
        features = self.model.features(x)
        out = torch.nn.functional.relu(features, inplace=True)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        features = torch.flatten(out, 1)
        return features


# ============================================================
# UTILITY FUNCTION: Load surrogate model from checkpoint
# ============================================================
def load_surrogate_model(model_type, checkpoint_path, num_classes=8, device='cuda'):
    """
    Load a trained surrogate model from checkpoint
    
    Args:
        model_type (str): 'vgg16', 'resnet50', 'efficientnet', or 'densenet'
        checkpoint_path (str): Path to the .pth checkpoint file
        num_classes (int): Number of output classes
        device (str): 'cuda' or 'cpu'
    
    Returns:
        model: Loaded model in eval mode
    """
    # Initialize model
    if model_type == 'vgg16':
        model = ModelA_VGG16(num_classes=num_classes, pretrained=False)
    elif model_type == 'resnet50':
        model = ModelAPrime_ResNet50(num_classes=num_classes, pretrained=False)
    elif model_type == 'efficientnet':
        model = ModelADoublePrime_EfficientNet(num_classes=num_classes, pretrained=False)
    elif model_type == 'densenet':
        model = ModelA_DenseNet121(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


# ============================================================
# TESTING FUNCTION
# ============================================================
if __name__ == "__main__":
    # Test model instantiation
    print("Testing Model A (VGG16)...")
    model_a = ModelA_VGG16(num_classes=8)
    dummy_input = torch.randn(2, 3, 224, 224)
    output_a = model_a(dummy_input)
    features_a = model_a.get_features(dummy_input)
    print(f"Model A output shape: {output_a.shape}")  # Should be [2, 8]
    print(f"Model A features shape: {features_a.shape}")
    
    print("\nTesting Model A' (ResNet-50)...")
    model_a_prime = ModelAPrime_ResNet50(num_classes=8)
    output_ap = model_a_prime(dummy_input)
    features_ap = model_a_prime.get_features(dummy_input)
    print(f"Model A' output shape: {output_ap.shape}")  # Should be [2, 8]
    print(f"Model A' features shape: {features_ap.shape}")
    
    print("\nTesting Model A'' (EfficientNet-B0)...")
    model_a_dp = ModelADoublePrime_EfficientNet(num_classes=8)
    output_adp = model_a_dp(dummy_input)
    features_adp = model_a_dp.get_features(dummy_input)
    print(f"Model A'' output shape: {output_adp.shape}")  # Should be [2, 8]
    print(f"Model A'' features shape: {features_adp.shape}")
    
    print("\nAll models instantiated successfully!")