import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms.v2 import ColorJitter, InterpolationMode

class PlacentaDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, subset_size=0, augment=False, augment_config=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.augment = augment
        
        # On-the-fly augmentation parameters
        if augment_config is None:
            augment_config = {
                'rotation_degrees': 30,
                'translation': (0.1, 0.1),
                'scale_range': (0.9, 1.1),
                'shear_degrees': 10,
                'brightness': 0.3,
                'contrast': 0.3,
                'saturation': 0.3,
                'hue': 0.1
            }
        self.augment_config = augment_config
        
        # Get all image and mask files
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.TIF', '.tif'))])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])
        
        # Create mapping of base names to full filenames
        self.image_map = {os.path.splitext(f)[0]: f for f in self.image_files}
        self.mask_map = {os.path.splitext(f)[0]: f for f in self.mask_files}
        
        # Get common base names
        self.common_names = sorted(set(self.image_map.keys()) & set(self.mask_map.keys()))
        
        assert len(self.common_names) > 0, "No matching image-mask pairs found!"
        
        # If subset_size > 0, only use the first subset_size images
        if subset_size and subset_size > 0:
            self.common_names = self.common_names[:subset_size]
            print(f"Using subset of {subset_size} images for training")
        
        # Print dataset info for debugging
        print(f"Found {len(self.common_names)} image-mask pairs")
        print(f"Image directory: {images_dir}")
        print(f"Mask directory: {masks_dir}")
        print(f"First few image files: {self.image_files[:5]}")
        print(f"First few mask files: {self.mask_files[:5]}")
    
    def __len__(self):
        return len(self.common_names)
    
    def __getitem__(self, idx):
        # Get base name and corresponding filenames
        base_name = self.common_names[idx]
        img_filename = self.image_map[base_name]
        mask_filename = self.mask_map[base_name]
        
        # Load image
        img_path = os.path.join(self.images_dir, img_filename)
        img = cv2.imread(img_path)  # shape: (H, W, 3) BGR
        
        # Check if image was loaded successfully
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
        
        # Load mask
        mask_path = os.path.join(self.masks_dir, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # shape: (H, W), values: 0, 1, 2
        
        # Check if mask was loaded successfully
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        # Convert image to float32 and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # For multi-class segmentation, keep mask as int64 (class indices)
        # Mask values should be 0 (background), 1 (fetal), 2 (maternal)
        mask = mask.astype(np.int64)
        
        # (Optional) transform: data augmentation, resizing, etc.
        if self.transform is not None:
            # e.g., if using Albumentations or custom transforms
            # sample = self.transform(image=img, mask=mask)
            # img, mask = sample['image'], sample['mask']
            pass
        
        # Convert to Torch Tensors
        # For segmentation, we typically have shape (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
        mask = torch.from_numpy(mask)    # (H, W) - CrossEntropyLoss expects this shape
        
        # On-the-fly augmentation
        if self.augment:
            img, mask = self._apply_augmentation(img, mask)
        
        return img, mask
    
    def _apply_augmentation(self, img, mask):
        """Apply random affine and color jitter augmentation."""
        cfg = self.augment_config
        
        # Random affine transformation
        angle = random.uniform(-cfg['rotation_degrees'], cfg['rotation_degrees'])
        max_dx = cfg['translation'][0] * img.shape[2]
        max_dy = cfg['translation'][1] * img.shape[1]
        translations = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))
        scale_factor = random.uniform(cfg['scale_range'][0], cfg['scale_range'][1])
        shear_angle = random.uniform(-cfg['shear_degrees'], cfg['shear_degrees'])
        
        # Apply affine to image
        img = TF.affine(
            img, angle=angle, translate=translations,
            scale=scale_factor, shear=shear_angle,
            interpolation=InterpolationMode.BILINEAR
        )
        
        # Apply same affine to mask
        mask_float = mask.float().unsqueeze(0)  # Add channel dimension
        mask_float = TF.affine(
            mask_float, angle=angle, translate=translations,
            scale=scale_factor, shear=shear_angle,
            interpolation=InterpolationMode.NEAREST
        )
        mask = mask_float.squeeze(0).round().to(torch.int64)
        
        # Apply color jitter only to image
        color_jitter = ColorJitter(
            brightness=cfg['brightness'],
            contrast=cfg['contrast'],
            saturation=cfg['saturation'],
            hue=cfg['hue']
        )
        img = color_jitter(img)
        
        return img, mask