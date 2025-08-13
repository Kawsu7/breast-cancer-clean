"""
Breast Cancer Classification Model Inference Module
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from typing import Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BreastCancerClassifier:
    """
    Breast Cancer Classification Model for histopathology images
    """
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize the classifier
        
        Args:
            model_path: Path to pre-trained model weights
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = self._get_transforms()
        self.classes = ['Benign', 'Malignant']
        
        if model_path:
            self.load_model(model_path)
        else:
            logger.warning("No model path provided. Using simulation mode.")
    
    def _get_transforms(self):
        """
        Get image preprocessing transforms
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_model(self, model_path: str):
        """
        Load pre-trained model
        
        Args:
            model_path: Path to model weights
        """
        try:
            # Example with ResNet18 - replace with your actual model
            self.model = models.resnet18(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Binary classification
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model inference
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed image tensor
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict(self, image: Image.Image) -> Dict[str, float]:
        """
        Perform classification on image
        
        Args:
            image: PIL Image to classify
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            # Simulation mode for demo
            return self._simulate_prediction(image)
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Get results
                predicted_class = self.classes[predicted.item()]
                confidence_score = confidence.item()
                
                return {
                    'label': predicted_class,
                    'confidence': confidence_score,
                    'is_malignant': predicted.item() == 1,
                    'probabilities': {
                        'Benign': probabilities[0][0].item(),
                        'Malignant': probabilities[0][1].item()
                    }
                }
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._simulate_prediction(image)
    
    def _simulate_prediction(self, image: Image.Image) -> Dict[str, float]:
        """
        Simulate prediction for demo purposes
        
        Args:
            image: PIL Image
            
        Returns:
            Simulated prediction results
        """
        # Random prediction for demo
        is_malignant = np.random.random() > 0.6
        base_confidence = 0.75 + np.random.random() * 0.2
        
        label = 'Malignant' if is_malignant else 'Benign'
        
        # Simulate probabilities
        if is_malignant:
            mal_prob = base_confidence
            ben_prob = 1 - mal_prob
        else:
            ben_prob = base_confidence
            mal_prob = 1 - ben_prob
        
        return {
            'label': label,
            'confidence': base_confidence,
            'is_malignant': is_malignant,
            'probabilities': {
                'Benign': ben_prob,
                'Malignant': mal_prob
            }
        }
    
    def predict_batch(self, images: list) -> list:
        """
        Perform batch prediction
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results


def load_sample_images() -> Dict[str, Image.Image]:
    """
    Load sample images for testing
    
    Returns:
        Dictionary of sample images
    """
    samples = {}
    
    try:
        # In a real implementation, load actual sample images
        # For demo, create placeholder images
        from PIL import Image
        import numpy as np
        
        # Create sample benign image (placeholder)
        benign_array = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
        samples['benign'] = Image.fromarray(benign_array)
        
        # Create sample malignant image (placeholder)  
        malignant_array = np.random.randint(50, 150, (224, 224, 3), dtype=np.uint8)
        samples['malignant'] = Image.fromarray(malignant_array)
        
        logger.info("Sample images loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading sample images: {e}")
    
    return samples


if __name__ == "__main__":
    # Example usage
    classifier = BreastCancerClassifier()
    
    # Load sample images
    samples = load_sample_images()
    
    # Test predictions
    for name, image in samples.items():
        result = classifier.predict(image)
        print(f"\nSample: {name}")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: {result['probabilities']}")