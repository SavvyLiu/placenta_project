#!/usr/bin/env python3
"""
Batch cleanup utility for mask images.
Converts any non-black pixels to white (255) in mask images.
Useful for normalizing mask formats before training.
"""

import os
import argparse
import logging
from pathlib import Path
from PIL import Image
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_image(path: str, backup: bool = False) -> bool:
    """
    Convert any non-black pixels to white in an image.
    
    Args:
        path: Path to the image file
        backup: If True, creates a backup before modifying
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if backup:
            backup_path = str(path) + '.backup'
            import shutil
            shutil.copy2(path, backup_path)
            logger.info(f"Created backup: {backup_path}")
        
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
        return True
    
    except Exception as e:
        logger.error(f"Error processing {path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Clean up mask images by converting non-black pixels to white"
    )
    parser.add_argument(
        '--dir', '-d',
        help='Directory containing mask images (default: data/masks)'
    )
    parser.add_argument(
        '--pattern',
        default='*.png',
        help='File pattern to match (default: *.png)'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backups of original files before modifying'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without making changes'
    )
    
    args = parser.parse_args()
    
    # Determine masks directory
    if args.dir:
        masks_dir = args.dir
    else:
        # Use default path relative to project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        masks_dir = os.path.join(project_dir, 'data', 'masks')
    
    # Verify directory exists
    if not os.path.isdir(masks_dir):
        logger.error(f"Directory does not exist: {masks_dir}")
        return 1
    
    # Find matching files
    files = list(Path(masks_dir).glob(args.pattern))
    
    if not files:
        logger.warning(f"No files matching '{args.pattern}' found in {masks_dir}")
        return 0
    
    logger.info(f"Found {len(files)} mask image(s) in {masks_dir}")
    
    if args.dry_run:
        logger.info("[DRY RUN] Would process the following files:")
        for f in files:
            logger.info(f"  {f.name}")
        return 0
    
    # Process files
    success_count = 0
    fail_count = 0
    
    for file_path in files:
        if process_image(str(file_path), backup=args.backup):
            logger.info(f"Processed: {file_path.name}")
            success_count += 1
        else:
            logger.error(f"Failed: {file_path.name}")
            fail_count += 1
    
    # Summary
    logger.info("-" * 50)
    logger.info(f"Processing complete:")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Failed:  {fail_count}")
    
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
