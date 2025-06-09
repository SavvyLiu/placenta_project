#!/usr/bin/env python3
"""
This script scans a directory of PNG mask images, converts any non-black pixels to white,
and overwrites the original files with the modified images.

Usage:
1. The masks folder is set via MASKS_DIR below.
2. Run:
   python cleanup_masks.py

All .png files in MASKS_DIR will be processed in place.
"""
import os
from PIL import Image
import numpy as np

# -------------------------------------
# Configuration: set your masks folder here
# -------------------------------------
# For example, if your masks are in:
#   ~/PLACENTA_FINAL/placenta_project/data/masks
# and this script lives in models/, then:
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASKS_DIR   = os.path.join(PROJECT_DIR, 'data', 'masks')

# -------------------------------------
# Process each PNG in the directory
# -------------------------------------
def process_image(path: str):
    img = Image.open(path)
    arr = np.array(img)

    # Grayscale: set any non-zero pixel to white (255)
    if arr.ndim == 2:
        arr[arr != 0] = 255
    else:
        # RGB or RGBA: mark any pixel where any channel != 0 as white
        if arr.shape[2] == 4:
            rgb   = arr[..., :3]
            alpha = arr[..., 3]
            mask  = np.any(rgb != 0, axis=2)
            rgb[mask] = [255, 255, 255]
            arr = np.dstack((rgb, alpha))
        else:
            mask = np.any(arr != 0, axis=2)
            arr[mask] = [255, 255, 255]

    # Save back to the same path
    Image.fromarray(arr).save(path)


def main():
    if not os.path.isdir(MASKS_DIR):
        print(f"Error: directory does not exist: {MASKS_DIR}")
        return

    files = [f for f in os.listdir(MASKS_DIR) if f.lower().endswith('.png')]
    if not files:
        print(f"No PNG files found in {MASKS_DIR}")
        return

    print(f"Processing {len(files)} mask images in {MASKS_DIR}...")
    for f in files:
        try:
            process_image(os.path.join(MASKS_DIR, f))
            print(f"  ✓ {f}")
        except Exception as e:
            print(f"  ✗ {f}: {e}")

    print("Done: non-black pixels have been set to white.")

if __name__ == '__main__':
    main()
