"""
Configuration module for inference and model settings.
"""

from dataclasses import dataclass
from typing import Tuple, Literal
import os


@dataclass
class InferenceConfig:
    """Configuration for inference operations"""
    
    # Model settings
    architecture: Literal['unet', 'efficientnet', 'regnet', 'vit'] = 'efficientnet'
    model_path: str = None  # If None, uses default path
    
    # Input/Output paths
    input_image: str = None
    output_mask: str = None
    output_annotated: str = None
    
    # Inference parameters
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    threshold: float = 0.5  # Classification threshold
    batch_size: int = 1
    
    # Post-processing
    morphology_kernel_size: Tuple[int, int] = (3, 3)
    morphology_operation: Literal['open', 'close', 'both', 'none'] = 'open'
    min_contour_area: int = 10
    
    # Visualization
    draw_contours: bool = True
    use_bounding_box: bool = False
    contour_color: Tuple[int, int, int] = (0, 255, 0)  # BGR
    contour_thickness: int = 2
    
    # Evaluation
    compute_metrics: bool = False
    ground_truth_path: str = None
    
    # Logging
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        valid_architectures = ['unet', 'efficientnet', 'regnet', 'vit']
        if self.architecture not in valid_architectures:
            raise ValueError(f"Invalid architecture: {self.architecture}. "
                           f"Must be one of {valid_architectures}")
        
        if self.threshold < 0 or self.threshold > 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {self.threshold}")
        
        if self.device not in ['auto', 'cuda', 'cpu']:
            raise ValueError(f"Device must be 'auto', 'cuda', or 'cpu', got {self.device}")


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation"""
    
    # Affine transformation parameters
    rotation_degrees: int = 30
    translation: Tuple[float, float] = (0.1, 0.1)  # Fraction of image size
    scale_range: Tuple[float, float] = (0.9, 1.1)
    shear_degrees: int = 10
    
    # Color jittering parameters
    brightness: float = 0.5
    contrast: float = 0.5
    saturation: float = 0.5
    hue: float = 0.2
    
    # Augmentation repetitions
    num_repetitions: int = 1
    
    # Output settings
    output_dir: str = None  # If None, saves to data/ directory
    prefix: str = 'augment'


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    
    # Hyperparameters
    num_epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-4
    
    # Scheduling
    lr_scheduler: Literal['reduce_on_plateau', 'none'] = 'reduce_on_plateau'
    lr_patience: int = 5
    lr_factor: float = 0.5
    
    # Dataset
    subset_size: int = 0  # 0 means use full dataset
    
    # Checkpointing
    save_interval: int = 100  # Save every N epochs
    save_dir: str = 'trained_models'
    
    # Device
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'


def load_config_from_file(config_path: str) -> dict:
    """
    Load configuration from a JSON or YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Dictionary of configuration values
    """
    import json
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}. "
                           "Supported: .json")


# Note: get_device() is available in utilities/model_loader.py
