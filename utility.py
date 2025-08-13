"""
Utility functions for breast cancer classification
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from typing import Dict, List, Tuple, Optional
import logging
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_uploaded_file(uploaded_file, upload_dir: str = "uploads") -> str:
    """
    Save uploaded file to disk
    
    Args:
        uploaded_file: Streamlit uploaded file object
        upload_dir: Directory to save files
        
    Returns:
        Path to saved file
    """
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    logger.info(f"File saved: {file_path}")
    return file_path

def validate_image(image_path: str) -> bool:
    """
    Validate if file is a valid image
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if valid image, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.error(f"Invalid image: {e}")
        return False

def resize_image(image: Image.Image, size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Resize image maintaining aspect ratio
    
    Args:
        image: PIL Image
        size: Target size (width, height)
        
    Returns:
        Resized image
    """
    # Calculate aspect ratio
    aspect_ratio = image.width / image.height
    target_aspect = size[0] / size[1]
    
    if aspect_ratio > target_aspect:
        # Image is wider
        new_width = size[0]
        new_height = int(size[0] / aspect_ratio)
    else:
        # Image is taller
        new_height = size[1]
        new_width = int(size[1] * aspect_ratio)
    
    # Resize and pad if necessary
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new image with target size and paste resized image
    new_image = Image.new('RGB', size, (255, 255, 255))
    paste_x = (size[0] - new_width) // 2
    paste_y = (size[1] - new_height) // 2
    new_image.paste(resized, (paste_x, paste_y))
    
    return new_image

def image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string
    
    Args:
        image: PIL Image
        
    Returns:
        Base64 encoded string
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def base64_to_image(base64_str: str) -> Image.Image:
    """
    Convert base64 string to PIL Image
    
    Args:
        base64_str: Base64 encoded image string
        
    Returns:
        PIL Image
    """
    img_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(img_data))
    return image

def calculate_confidence_metrics(predictions: List[Dict]) -> Dict:
    """
    Calculate aggregate confidence metrics
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        Metrics dictionary
    """
    confidences = [pred['confidence'] for pred in predictions]
    
    metrics = {
        'mean_confidence': np.mean(confidences),
        'median_confidence': np.median(confidences),
        'std_confidence': np.std(confidences),
        'min_confidence': np.min(confidences),
        'max_confidence': np.max(confidences),
        'high_confidence_count': sum(1 for c in confidences if c > 0.8),
        'low_confidence_count': sum(1 for c in confidences if c < 0.6)
    }
    
    return metrics

def format_prediction_result(result: Dict) -> str:
    """
    Format prediction result for display
    
    Args:
        result: Prediction result dictionary
        
    Returns:
        Formatted string
    """
    label = result['label']
    confidence = result['confidence'] * 100
    
    emoji = "🔴" if result['is_malignant'] else "🟢"
    
    formatted = f"{emoji} **{label}** ({confidence:.1f}% confidence)"
    
    if result['is_malignant']:
        formatted += "\n⚠️ *Requires medical attention*"
    else:
        formatted += "\n✅ *Regular monitoring recommended*"
    
    return formatted

def create_sample_dataset(num_samples: int = 10) -> List[Dict]:
    """
    Create sample dataset for testing
    
    Args:
        num_samples: Number of sample images to create
        
    Returns:
        List of sample data dictionaries
    """
    samples = []
    
    for i in range(num_samples):
        # Alternate between benign and malignant
        is_malignant = i % 2 == 0
        label = "Malignant" if is_malignant else "Benign"
        
        # Create random image data
        if is_malignant:
            # Darker, more irregular patterns for malignant
            img_array = np.random.randint(30, 120, (224, 224, 3), dtype=np.uint8)
        else:
            # Lighter, more regular patterns for benign
            img_array = np.random.randint(120, 220, (224, 224, 3), dtype=np.uint8)
        
        image = Image.fromarray(img_array)
        
        sample = {
            'id': f"sample_{i:03d}",
            'image': image,
            'label': label,
            'is_malignant': is_malignant,
            'filename': f"sample_{label.lower()}_{i:03d}.jpg"
        }
        
        samples.append(sample)
    
    return samples

def log_prediction(result: Dict, metadata: Dict = None) -> None:
    """
    Log prediction result for monitoring
    
    Args:
        result: Prediction result
        metadata: Additional metadata
    """
    log_entry = {
        'timestamp': np.datetime64('now').isoformat(),
        'prediction': result['label'],
        'confidence': result['confidence'],
        'is_malignant': result['is_malignant']
    }
    
    if metadata:
        log_entry.update(metadata)
    
    logger.info(f"Prediction logged: {json.dumps(log_entry)}")

def get_model_info() -> Dict:
    """
    Get model information and system specs
    
    Returns:
        Model and system information
    """
    info = {
        'model_type': 'ResNet18 (Demo)',
        'input_size': '224x224',
        'classes': ['Benign', 'Malignant'],
        'framework': 'PyTorch',
        'device': 'CUDA' if torch.cuda.is_available() else 'CPU',
        'python_version': '3.8+',
        'streamlit_version': '1.28.1'
    }
    
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
    
    return info

def cleanup_temp_files(directory: str = "uploads", max_age_hours: int = 24) -> None:
    """
    Clean up temporary uploaded files
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age of files to keep
    """
    import time
    
    if not os.path.exists(directory):
        return
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getctime(file_path)
            
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up old file: {file_path}")
                except Exception as e:
                    logger.error(f"Error cleaning file {file_path}: {e}")

class ImageAugmentor:
    """
    Image augmentation utilities for model training/testing
    """
    
    @staticmethod
    def apply_random_rotation(image: Image.Image, max_angle: int = 15) -> Image.Image:
        """Apply random rotation to image"""
        angle = np.random.uniform(-max_angle, max_angle)
        return image.rotate(angle, fillcolor=(255, 255, 255))
    
    @staticmethod
    def apply_random_brightness(image: Image.Image, factor_range: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
        """Apply random brightness adjustment"""
        from PIL import ImageEnhance
        factor = np.random.uniform(*factor_range)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def apply_random_contrast(image: Image.Image, factor_range: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
        """Apply random contrast adjustment"""
        from PIL import ImageEnhance
        factor = np.random.uniform(*factor_range)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

# Export key functions
__all__ = [
    'save_uploaded_file',
    'validate_image', 
    'resize_image',
    'image_to_base64',
    'base64_to_image',
    'calculate_confidence_metrics',
    'format_prediction_result',
    'create_sample_dataset',
    'log_prediction',
    'get_model_info',
    'cleanup_temp_files',
    'ImageAugmentor'
]