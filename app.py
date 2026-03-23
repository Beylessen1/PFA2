from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import io
import os
from PIL import Image
import base64
import torchvision.transforms as transforms
import torchvision.models as models

app = Flask(__name__)
CORS(app)

# Define your CNN architecture - MATCHES TRAINING CODE
HIDDEN_LAYER_SIZE = 1000

class MalwareClassifier(nn.Module):
    def __init__(self, n_classes=8):
        super(MalwareClassifier, self).__init__()
        # Load pretrained ResNet50
        self.resnet = models.resnet50(weights=None)  # weights loaded from checkpoint
        
        # Freeze ResNet parameters
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Replace the last fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, n_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# Class labels (8 classes) - ORDER MATCHES train_dataset.classes
CLASS_LABELS = {
    0: "Adware",
    1: "Backdoor", 
    2: "Benign",
    3: "Downloader",
    4: "Spyware",
    5: "Trojan",
    6: "Virus",
    7: "Worm"
}

# Image preprocessing transform - MATCHES TRAINING CODE EXACTLY
def get_image_transform():
    """Get the image preprocessing pipeline - matches training"""
    return transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])

def preprocess_image(image_data):
    """
    Preprocess image for model input
    Args:
        image_data: PIL Image or base64 string or bytes
    Returns:
        Preprocessed tensor
    """
    # Handle different input types
    if isinstance(image_data, str):
        # Base64 string
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
    elif isinstance(image_data, bytes):
        # Raw bytes
        image = Image.open(io.BytesIO(image_data))
    else:
        # Assume PIL Image
        image = image_data
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    transform = get_image_transform()
    image_tensor = transform(image)
    
    return image_tensor

# Global model variable
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path='malware_classifier.pth'):
    """Load the trained model"""
    global model
    
    try:
        # Initialize model
        model = MalwareClassifier(n_classes=8)
        
        # Load the TorchScript model
        model = torch.jit.load(model_path, map_location=device)
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
        print(f"Model config: 8 classes, ResNet50 backbone, 224x224 input size")
        return True
        
    except Exception as e:
        print(f"Error loading TorchScript model: {str(e)}")
        print("Trying to load as state dict...")
        
        try:
            # Fallback: try loading as state dict
            model = MalwareClassifier(n_classes=8)

            #CORRECTION BY BANDIT
            checkpoint = torch.load(
                model_path,
                map_location=device,
                weights_only=True
            )
            model.load_state_dict(checkpoint)
            #END OF CORRECTION BY BANDIT
            
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            
            print(f"Model loaded successfully on {device}")
            print(f"Model config: 8 classes, ResNet50 backbone, 224x224 input size")
            return True
            
        except Exception as e2:
            print(f"Error loading model: {str(e2)}")
            return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

@app.route('/api/classify', methods=['POST'])
def classify():
    """Classify a malware image"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Check if it's a file upload or JSON data
        if 'image' in request.files:
            # Handle file upload
            file = request.files['image']
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
        elif request.is_json:
            # Handle JSON with base64 image
            data = request.get_json()
            
            if 'image' not in data:
                return jsonify({'error': 'No image provided'}), 400
            
            image_data = data['image']
            image = None
            
            # Handle base64 image
            if isinstance(image_data, str):
                if 'base64,' in image_data:
                    image_data = image_data.split('base64,')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                return jsonify({'error': 'Invalid image format'}), 400
        else:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess image
        input_tensor = preprocess_image(image)
        input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dimension
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get all class probabilities
        class_probabilities = {
            CLASS_LABELS[i]: float(probabilities[0][i].item())
            for i in range(len(CLASS_LABELS))
        }
        
        # Determine threat level
        prediction_name = CLASS_LABELS[predicted_class]
        if prediction_name == "Benign":
            threat_level = "Safe"
            threat_color = "green"
        elif prediction_name in ["Adware", "Downloader"]:
            threat_level = "Low"
            threat_color = "yellow"
        elif prediction_name in ["Spyware", "Trojan", "Worm"]:
            threat_level = "High"
            threat_color = "orange"
        else:  # Virus, Backdoor
            threat_level = "Critical"
            threat_color = "red"
        
        return jsonify({
            'prediction': prediction_name,
            'confidence': float(confidence),
            'threat_level': threat_level,
            'threat_color': threat_color,
            'probabilities': class_probabilities,
            'timestamp': request.json.get('timestamp', None) if request.is_json else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-classify', methods=['POST'])
def batch_classify():
    """Classify multiple image samples"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'samples' not in data:
            return jsonify({'error': 'No samples provided'}), 400
        
        results = []
        
        for idx, sample in enumerate(data['samples']):
            # Preprocess image
            image = preprocess_image(sample['image'])
            input_tensor = image.unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            results.append({
                'id': sample.get('id', idx),
                'prediction': CLASS_LABELS[predicted_class],
                'confidence': float(confidence)
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'classes': CLASS_LABELS,
        'num_classes': len(CLASS_LABELS),
        'device': str(device),
        'architecture': 'ResNet50 (Transfer Learning)',
        'input_type': 'RGB Image',
        'img_size': 224,
        'img_channels': 3,
        'normalization': 'ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])'
    })

if __name__ == '__main__':
    # Load the model on startup
    model_path = os.getenv('MODEL_PATH', 'malware_classifier.pth')
    
    if os.path.exists(model_path):
        load_model(model_path)
    else:
        print(f"Warning: Model file '{model_path}' not found!")
        print("Please place your model file in the same directory or set MODEL_PATH environment variable")
    
    # Run the server

    port = int(os.getenv("PORT", 5000))
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    app.run(host=host, port=port, debug=debug)
