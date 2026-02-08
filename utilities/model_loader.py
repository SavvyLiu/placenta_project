"""
Unified model loading module for placenta segmentation models.
Supports: U-Net (smp), EfficientNet, RegNet, Vision Transformer
All models output 3 classes: Background (0), Fetal (1), Maternal (2)
"""

import os
import torch
import logging
from typing import Literal
from models.efficicentnet_train_smp import EfficientNetUNet
from models.regnet_train_smp import RegNetUNet
from models.ViT_train_smp import ViT_UNet_Flexible
import segmentation_models_pytorch as smp


logger = logging.getLogger(__name__)


# Class indices for 3-class segmentation
CLASS_LABELS = {
    0: 'background',
    1: 'fetal',
    2: 'maternal'
}

# Color mapping for visualization (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 0, 0),              # Black for background
    1: (0, 255, 0),            # Green for fetal
    2: (0, 0, 255)             # Red for maternal
}


def get_model(
    architecture: Literal['unet', 'efficientnet', 'regnet', 'vit'],
    device: torch.device = None,
    n_classes: int = 3
) -> torch.nn.Module:
    """
    Instantiate a model of the specified architecture.
    
    Args:
        architecture: One of ['unet', 'efficientnet', 'regnet', 'vit']
        device: Device to place model on (cuda/cpu)
        n_classes: Number of output classes (default 3)
    
    Returns:
        Instantiated model on the specified device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if architecture == 'unet':
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes
        )
    elif architecture == 'efficientnet':
        model = EfficientNetUNet(n_classes=n_classes)
    elif architecture == 'regnet':
        model = RegNetUNet(n_classes=n_classes)
    elif architecture == 'vit':
        model = ViT_UNet_Flexible(n_classes=n_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        f"Choose from: unet, efficientnet, regnet, vit")
    
    model.to(device)
    model.eval()
    return model


def load_model(
    architecture: Literal['unet', 'efficientnet', 'regnet', 'vit'],
    model_path: str = None,
    device: torch.device = None,
    n_classes: int = 3
) -> torch.nn.Module:
    """
    Load a pretrained model from disk.
    
    Args:
        architecture: One of ['unet', 'efficientnet', 'regnet', 'vit']
        model_path: Path to the saved .pth file. If None, uses default path.
        device: Device to load model on (cuda/cpu)
        n_classes: Number of output classes (default 3)
    
    Returns:
        Loaded model on the specified device
    
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # If no path provided, use default naming convention
    if model_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_names = {
            'unet': 'smp_unet_placenta.pth',
            'efficientnet': 'efficientnet_unet_placenta.pth',
            'regnet': 'regnet_unet_placenta.pth',
            'vit': 'vit_unet_placenta_flexible.pth'
        }
        model_path = os.path.join(project_root, 'trained_models', default_names[architecture])
    
    # Check if file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model architecture
    model = get_model(architecture, device=device, n_classes=n_classes)
    
    # Load weights
    logger.info(f"Loading {architecture} model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully")
    
    return model


def get_default_model_path(architecture: str, project_root: str = None) -> str:
    """
    Get the default save path for a model architecture.
    
    Args:
        architecture: One of ['unet', 'efficientnet', 'regnet', 'vit']
        project_root: Project root directory. If None, inferred from utilities location.
    
    Returns:
        Path to default model file
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    default_names = {
        'unet': 'smp_unet_placenta.pth',
        'efficientnet': 'efficientnet_unet_placenta.pth',
        'regnet': 'regnet_unet_placenta.pth',
        'vit': 'vit_unet_placenta_flexible.pth'
    }
    
    if architecture not in default_names:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return os.path.join(project_root, 'trained_models', default_names[architecture])
