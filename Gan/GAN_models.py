"""
GAN Architecture for Adversarial Attack Generation
Implements Generator and Discriminator for AdvGAN-style attacks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# GENERATOR NETWORK
# ============================================================
class Generator(nn.Module):
    """
    Generator network that produces adversarial perturbations
    
    Architecture:
    - Takes (image, target_class) as input
    - Target class is embedded and concatenated with image features
    - Uses U-Net style encoder-decoder with skip connections
    - Outputs perturbation of same shape as input image
    
    The perturbation is added to the original image to create adversarial example
    """
    
    def __init__(self, num_classes=8, latent_dim=100, channels=[64, 128, 256, 512]):
        """
        Args:
            num_classes (int): Number of classes for target conditioning
            latent_dim (int): Dimension of target class embedding
            channels (list): Number of channels at each encoder/decoder level
        """
        super(Generator, self).__init__()
        
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # Target class embedding
        # Maps target class to a latent vector
        self.target_embedding = nn.Embedding(num_classes, latent_dim)
        
        # We'll reshape the embedding to add as extra channels
        # Embedding -> [B, latent_dim] -> [B, latent_dim, 1, 1] -> broadcast to [B, latent_dim, H, W]
        
        # ENCODER (Downsampling path)
        # Input: [B, 3+latent_dim, 224, 224] (image + embedded target broadcasted)
        self.enc1 = self._encoder_block(3 + latent_dim, channels[0])  # -> [B, 64, 112, 112]
        self.enc2 = self._encoder_block(channels[0], channels[1])      # -> [B, 128, 56, 56]
        self.enc3 = self._encoder_block(channels[1], channels[2])      # -> [B, 256, 28, 28]
        self.enc4 = self._encoder_block(channels[2], channels[3])      # -> [B, 512, 14, 14]
        
        # BOTTLENECK
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[3], channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True)
        )
        
        # DECODER (Upsampling path with skip connections)
        self.dec4 = self._decoder_block(channels[3], channels[2])      # -> [B, 256, 28, 28]
        self.dec3 = self._decoder_block(channels[2] * 2, channels[1])  # *2 for skip connection
        self.dec2 = self._decoder_block(channels[1] * 2, channels[0])
        self.dec1 = self._decoder_block(channels[0] * 2, channels[0])
        
        # Final output layer
        # Output: [B, 3, 224, 224] - perturbation in image space
        self.final = nn.Sequential(
            nn.Conv2d(channels[0], 3, kernel_size=1),
            nn.Tanh()  # Perturbation in range [-1, 1]
        )
    
    def _encoder_block(self, in_channels, out_channels):
        """Create an encoder block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def _decoder_block(self, in_channels, out_channels):
        """Create a decoder block: Upsample -> Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, target_class):
        """
        Forward pass
        
        Args:
            x: Input images [B, 3, H, W]
            target_class: Target class labels [B] (integers)
        
        Returns:
            perturbation: Generated perturbation [B, 3, H, W] in range [-1, 1]
        """
        batch_size = x.size(0)
        
        # Embed target class: [B] -> [B, latent_dim]
        target_embed = self.target_embedding(target_class)
        
        # Reshape to add as spatial feature maps: [B, latent_dim] -> [B, latent_dim, H, W]
        target_embed = target_embed.view(batch_size, self.latent_dim, 1, 1)
        target_embed = target_embed.expand(batch_size, self.latent_dim, x.size(2), x.size(3))
        
        # Concatenate image and target embedding along channel dimension
        x_with_target = torch.cat([x, target_embed], dim=1)  # [B, 3+latent_dim, H, W]
        
        # Encoder path (save features for skip connections)
        enc1_out = self.enc1(x_with_target)  # [B, 64, 112, 112]
        enc2_out = self.enc2(enc1_out)        # [B, 128, 56, 56]
        enc3_out = self.enc3(enc2_out)        # [B, 256, 28, 28]
        enc4_out = self.enc4(enc3_out)        # [B, 512, 14, 14]
        
        # Bottleneck
        bottleneck_out = self.bottleneck(enc4_out)
        
        # Decoder path with skip connections
        dec4_out = self.dec4(bottleneck_out)
        dec4_out = torch.cat([dec4_out, enc3_out], dim=1)  # Skip connection
        
        dec3_out = self.dec3(dec4_out)
        dec3_out = torch.cat([dec3_out, enc2_out], dim=1)  # Skip connection
        
        dec2_out = self.dec2(dec3_out)
        dec2_out = torch.cat([dec2_out, enc1_out], dim=1)  # Skip connection
        
        dec1_out = self.dec1(dec2_out)
        
        # Final output: perturbation in range [-1, 1]
        perturbation = self.final(dec1_out)
        
        return perturbation


# ============================================================
# DISCRIMINATOR NETWORK
# ============================================================
class Discriminator(nn.Module):
    """
    Discriminator network that classifies images as real or adversarial
    
    Architecture:
    - Takes an image as input
    - Uses convolutional layers to extract features
    - Outputs a single value (real/fake probability)
    """
    
    def __init__(self, channels=[64, 128, 256, 512]):
        """
        Args:
            channels (list): Number of channels at each convolutional level
        """
        super(Discriminator, self).__init__()
        
        # Convolutional layers
        # Input: [B, 3, 224, 224]
        self.conv1 = self._conv_block(3, channels[0], normalize=False)  # [B, 64, 112, 112]
        self.conv2 = self._conv_block(channels[0], channels[1])          # [B, 128, 56, 56]
        self.conv3 = self._conv_block(channels[1], channels[2])          # [B, 256, 28, 28]
        self.conv4 = self._conv_block(channels[2], channels[3])          # [B, 512, 14, 14]
        
        # Global average pooling + final classification
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # [B, 512, 1, 1]
        self.fc = nn.Sequential(
            nn.Linear(channels[3], 1),
            nn.Sigmoid()  # Output probability in [0, 1]
        )
    
    def _conv_block(self, in_channels, out_channels, normalize=True):
        """Create a convolutional block: Conv -> (BN) -> LeakyReLU"""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            output: Probability that image is real (not adversarial) [B, 1]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.global_pool(x)  # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 512]
        
        output = self.fc(x)  # [B, 1]
        
        return output


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def weights_init(m):
    """
    Initialize network weights
    Convention from DCGAN paper
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def test_generator(num_classes=8, batch_size=4):
    """Test generator forward pass"""
    print("\nTesting Generator...")
    generator = Generator(num_classes=num_classes)
    generator.apply(weights_init)
    
    # Create dummy input
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_targets = torch.randint(0, num_classes, (batch_size,))
    
    # Forward pass
    perturbations = generator(dummy_images, dummy_targets)
    
    print(f"Input shape: {dummy_images.shape}")
    print(f"Target classes: {dummy_targets}")
    print(f"Perturbation shape: {perturbations.shape}")
    print(f"Perturbation range: [{perturbations.min():.3f}, {perturbations.max():.3f}]")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    return generator


def test_discriminator(batch_size=4):
    """Test discriminator forward pass"""
    print("\nTesting Discriminator...")
    discriminator = Discriminator()
    discriminator.apply(weights_init)
    
    # Create dummy input
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    outputs = discriminator(dummy_images)
    
    print(f"Input shape: {dummy_images.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    return discriminator


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("Testing GAN Architecture")
    print("="*60)
    
    # Test Generator
    generator = test_generator(num_classes=8, batch_size=4)
    
    # Test Discriminator
    discriminator = test_discriminator(batch_size=4)
    
    print("\n" + "="*60)
    print("GAN Architecture Test Complete!")
    print("="*60)