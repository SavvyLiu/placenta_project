import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class PlacentaDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, subset_size=0):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
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
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # shape: (H, W)
        
        # Check if mask was loaded successfully
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        # Convert to numpy float32, scale to [0,1]
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        
        # (Optional) transform: data augmentation, resizing, etc.
        if self.transform is not None:
            # e.g., if using Albumentations or custom transforms
            # sample = self.transform(image=img, mask=mask)
            # img, mask = sample['image'], sample['mask']
            pass
        
        # Convert to Torch Tensors
        # For segmentation, we typically have shape (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)    # (1, H, W)
        
        return img, mask