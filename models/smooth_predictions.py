#!/usr/bin/env python3
"""
This script applies morphological closing and opening to smooth the binary mask predictions
produced by your inference script. It processes all .png files in the predictions folder,
overwrites them in place, and preserves the original mask areas while rounding jagged edges.

Usage:
1. Place this file in your models/ directory (next to infer_efficientnet.py).
2. Adjust KERNEL_SIZE or the predictions folder path below if needed.
3. Run:
   python smooth_predictions.py
"""
import os
import glob
import cv2

# -------------------------------------
# Configuration
# -------------------------------------
# Path to your predictions folder (from inference script)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR    = os.path.join(PROJECT_DIR, 'data', 'validation', 'predictions')

# Morphological kernel size (must be odd)
KERNEL_SIZE = 7  # e.g. 3, 5, 7 for increasing smoothing

# -------------------------------------
# Build structuring element
# -------------------------------------
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))

# -------------------------------------
# Process each PNG
# -------------------------------------
if not os.path.isdir(PRED_DIR):
    print(f"Error: predictions directory not found: {PRED_DIR}")
    exit(1)

files = glob.glob(os.path.join(PRED_DIR, '*.png'))
if not files:
    print(f"No PNG files found in {PRED_DIR}")
    exit(0)

print(f"Smoothing {len(files)} prediction masks in {PRED_DIR} with kernel size {KERNEL_SIZE}...")
for path in files:
    # Read as grayscale binary mask
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        print(f"  ✗ failed to read {path}")
        continue

    # Ensure binary values 0 or 255
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

    # Morphological closing then opening
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel)

    # Overwrite the original
    cv2.imwrite(path, m)
    print(f"  ✓ {os.path.basename(path)}")

print("Smoothing complete.")
