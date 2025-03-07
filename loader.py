import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class PlacentaDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        self.image_names = sorted(os.listdir(images_dir))
        self.mask_names = sorted(os.listdir(masks_dir))
        
        assert len(self.image_names) == len(self.mask_names), \
            "Number of images and masks must match!"
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_names[idx])
        img = cv2.imread(img_path)  # shape: (H, W, 3) BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
        
        # Load mask
        mask_path = os.path.join(self.masks_dir, self.mask_names[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # shape: (H, W)
        
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