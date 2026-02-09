#!/usr/bin/env python
"""
Diagnostic script to check mask formats and values
"""
import cv2
import os
from pathlib import Path
import numpy as np

mask_dir = Path("data/masks")
masks = list(mask_dir.glob("*.png"))[:5]  # Check first 5

print("=" * 70)
print("MASK DIAGNOSTIC")
print("=" * 70)

for mask_path in masks:
    # Read as grayscale
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Read as color
    mask_color = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    
    print(f"\n{mask_path.name}:")
    
    if mask_gray is not None:
        print(f"  Grayscale shape: {mask_gray.shape}")
        print(f"  Grayscale unique values: {np.unique(mask_gray)}")
        print(f"  Grayscale min/max: {mask_gray.min()}/{mask_gray.max()}")
    
    if mask_color is not None:
        print(f"  Color shape: {mask_color.shape}")
        if len(mask_color.shape) == 3:
            print(f"  Color - Channel 0 unique: {np.unique(mask_color[:,:,0])}")
            print(f"  Color - Channel 1 unique: {np.unique(mask_color[:,:,1])}")
            print(f"  Color - Channel 2 unique: {np.unique(mask_color[:,:,2])}")

print("\n" + "=" * 70)
print("If all channels have the same values, masks are grayscale (saved in RGB).")
print("If channels differ, masks are actually RGB and need processing.")
print("=" * 70)
