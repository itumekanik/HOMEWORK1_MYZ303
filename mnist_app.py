import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import os
import sys

# Neural Network architecture (same as training model)
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # Input: 28x28 = 784 features
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)  # 10 output classes (digits 0-9)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

def preprocess_image(image_path):
    """Preprocess the input image to match MNIST format"""
    try:
        # Open image and convert to grayscale
        img = Image.open(image_path).convert('L')
        
        # Resize to 28x28
        img = img.resize((28, 28))
        
        # Check if we need to invert the image
        img_array = np.array(img)
        mean_pixel = np.mean(img_array)
        if mean_pixel > 128:  # If background is bright, invert
            img = Image.fromarray(255 - img_array)
        
        # Apply transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST dataset mean and std
        ])
        
        # Transform and add batch dimension
        tensor = transform(img).unsqueeze(0)
        
        return tensor
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)

def load_model(model_path):
    """Load the trained model"""
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found!")
            sys.exit(1)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MNISTNet().to(device)
        
        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set to evaluation mode
        
        return model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def predict_digit(model, img_tensor, device):
    """Make a prediction using the model and return statistics for all 10 categories"""
    try:
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            # Convert log probabilities to probabilities
            probabilities = torch.exp(output).cpu().numpy()[0]
            pred = int(probabilities.argmax())
            confidence = probabilities[pred] * 100
            
        return pred, confidence, probabilities
    except Exception as e:
        print(f"Error making prediction: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='MNIST Digit Recognition')
    parser.add_argument('image', type=str, help='Path to input image file')
    parser.add_argument('--model', type=str, default='models/mnist_ann_model.pth', 
                        help='Path to trained model file (default: models/mnist_ann_model.pth)')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Show detailed output with confidence score')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        sys.exit(1)
    
    # Load the model
    model, device = load_model(args.model)
    
    # Preprocess the image
    img_tensor = preprocess_image(args.image)
    
    # Make prediction and get evaluation statistics for all 10 categories
    prediction, confidence, probabilities = predict_digit(model, img_tensor, device)
    
    # Print evaluation statistics for 10 categories
    print("Evaluation Statistics for 10 Categories:")
    for digit, prob in enumerate(probabilities):
        print(f"Digit {digit}: {prob * 100:.2f}%")
    
    # Print the final predicted digit separately
    print(f"\nPredicted digit: {prediction} (Confidence: {confidence:.2f}%)")

if __name__ == "__main__":
    main()
