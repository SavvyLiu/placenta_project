"""
Utility for converting colored masks to binary or multi-class masks.
Supports: Grey foreground detection, custom threshold conversion
"""

import cv2
import numpy as np
import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_grey_foreground_mask(mask_path: str, output_binary_path: str) -> np.ndarray:
    """
    Converts a colored mask with a grey foreground and a black background
    into a binary mask where the grey regions are white (255) and the background is black (0).

    Parameters:
      - mask_path: Path to the input colored mask image.
      - output_binary_path: Path to save the resulting binary mask.

    Returns:
      - binary_mask: The binary mask as a numpy array.
    """
    # Verify input file exists
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    # Read the mask image (BGR format)
    mask = cv2.imread(mask_path)
    if mask is None:
        raise ValueError(f"Could not read image: {mask_path}")
    
    logger.info(f"Loaded mask: {mask_path}, shape: {mask.shape}")

    # Convert to HSV color space
    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    # Define thresholds for grey (low saturation and mid-range brightness)
    lower_grey = np.array([0, 0, 50])  # Low saturation, mid brightness
    upper_grey = np.array([180, 50, 200])  # Allow slight color variations but keep low saturation

    # Create a mask for grey regions
    grey_mask = cv2.inRange(hsv, lower_grey, upper_grey)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_binary_path), exist_ok=True)
    
    cv2.imwrite(output_binary_path, grey_mask)
    logger.info(f"Binary mask saved to {output_binary_path}")
    return grey_mask


def convert_by_threshold(
    mask_path: str,
    output_path: str,
    threshold: int = 127
) -> np.ndarray:
    """
    Convert a grayscale image to binary using a simple threshold.
    
    Parameters:
        mask_path: Path to input image
        output_path: Path to save output
        threshold: Threshold value (0-255)
    
    Returns:
        Binary mask
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read image: {mask_path}")
    
    logger.info(f"Loaded mask: {mask_path}")
    
    # Apply threshold
    _, binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, binary_mask)
    logger.info(f"Converted mask saved to {output_path}")
    return binary_mask


def main():
    parser = argparse.ArgumentParser(
        description="Convert masks from various formats to binary or multi-class representations"
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input mask image(s) or directory'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Path to save output mask(s)'
    )
    parser.add_argument(
        '--method',
        choices=['grey', 'threshold'],
        default='threshold',
        help='Conversion method (default: threshold)'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=127,
        help='Threshold value for threshold method (0-255, default: 127)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all images in input directory'
    )
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            # Process directory
            if not os.path.isdir(args.input):
                raise ValueError(f"--batch requires input to be a directory, got: {args.input}")
            
            os.makedirs(args.output, exist_ok=True)
            image_files = list(Path(args.input).glob('*.png')) + list(Path(args.input).glob('*.jpg'))
            
            logger.info(f"Processing {len(image_files)} images from {args.input}")
            
            for img_path in image_files:
                try:
                    output_file = os.path.join(args.output, img_path.name)
                    
                    if args.method == 'grey':
                        convert_grey_foreground_mask(str(img_path), output_file)
                    else:
                        convert_by_threshold(str(img_path), output_file, args.threshold)
                    
                    logger.info(f"  Processed: {img_path.name}")
                except Exception as e:
                    logger.error(f"  Error processing {img_path.name}: {e}")
        else:
            # Process single image
            if args.method == 'grey':
                convert_grey_foreground_mask(args.input, args.output)
            else:
                convert_by_threshold(args.input, args.output, args.threshold)
        
        logger.info("Conversion completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
