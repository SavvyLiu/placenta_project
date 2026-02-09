import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision.transforms.v2 import ColorJitter, InterpolationMode
import torchvision.transforms.functional as TF

# Optional: Import config for programmatic usage
try:
    from utilities.config import AugmentationConfig
except ImportError:
    AugmentationConfig = None


# -------------------------------------------------------------------
# Custom Joint Random Affine Transformation
# -------------------------------------------------------------------
class JointRandomAffine:
    """
    Compute random affine parameters (rotation, translation, scale, shear)
    and apply the transformation to both image and mask using the same parameters.
    The mask is temporarily converted to float (since grid_sample does not support int64)
    and then rounded back to int.
    """

    def __init__(self, degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        # Sample a random rotation angle.
        angle = random.uniform(-self.degrees, self.degrees)

        # Sample random translations (in pixels) based on image dimensions.
        if self.translate is not None:
            max_dx = self.translate[0] * image.shape[2]  # width
            max_dy = self.translate[1] * image.shape[1]  # height
            translations = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))
        else:
            translations = (0, 0)

        # Random scaling factor.
        if self.scale is not None:
            scale_factor = random.uniform(self.scale[0], self.scale[1])
        else:
            scale_factor = 1.0

        # Random shear angle.
        if self.shear is not None:
            shear_angle = random.uniform(-self.shear, self.shear)
        else:
            shear_angle = 0.0

        # Apply affine transformation to the image.
        image_trans = TF.affine(
            image, angle=angle, translate=translations,
            scale=scale_factor, shear=shear_angle,
            interpolation=InterpolationMode.BILINEAR
        )

        # For the mask, add a channel dimension if needed.
        # (The mask is expected to be [1, H, W].)
        mask_float = mask.float()
        mask_trans = TF.affine(
            mask_float, angle=angle, translate=translations,
            scale=scale_factor, shear=shear_angle,
            interpolation=InterpolationMode.NEAREST
        )
        mask_trans = mask_trans.round().to(torch.int64)
        return image_trans, mask_trans


# -------------------------------------------------------------------
# Custom Joint Augmentation combining JointRandomAffine with ColorJitter
# -------------------------------------------------------------------
class CustomJointAugment:
    def __init__(self, affine_params=None, color_jitter_params=None, config=None):
        """
        Initialize joint augmentation.
        
        Args:
            affine_params: Dict of affine parameters (used if config is None)
            color_jitter_params: Dict of color jitter parameters (used if config is None)
            config: Optional AugmentationConfig instance (takes precedence)
        """
        if config is not None and AugmentationConfig is not None:
            # Use config if provided
            affine_params = {
                "degrees": config.rotation_degrees,
                "translate": config.translation,
                "scale": config.scale_range,
                "shear": config.shear_degrees
            }
            color_jitter_params = {
                "brightness": config.brightness,
                "contrast": config.contrast,
                "saturation": config.saturation,
                "hue": config.hue
            }
        elif affine_params is None:
            affine_params = {"degrees": 30, "translate": (0.1, 0.1), "scale": (0.9, 1.1), "shear": 10}
        
        self.joint_affine = JointRandomAffine(**affine_params)
        
        if color_jitter_params is None:
            # Adjust these parameters to get desired photometric variability.
            color_jitter_params = {"brightness": 0.5, "contrast": 0.5, "saturation": 0.5, "hue": 0.2}
        self.color_jitter = ColorJitter(**color_jitter_params)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        # Apply the same affine transformation to both image and mask.
        image, mask = self.joint_affine(image, mask)
        # Apply color jitter only to the image.
        image = self.color_jitter(image)
        return image, mask


# Use the custom joint augmentation.
joint_transform = CustomJointAugment()


# -------------------------------------------------------------------
# Dataset class for segmentation tasks that reads files from folders.
# -------------------------------------------------------------------
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        image_dir: Path to the folder containing images.
        mask_dir: Path to the folder containing masks.
        transform: Callable taking (image, mask) and returning augmented versions.
        """
        # List image files (.png, .jpg, .tif, .tiff)
        image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        self.image_paths = sorted([os.path.join(image_dir, f)
                                   for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)])
        
        # List mask files (.png only, as per standard)
        self.mask_paths = sorted([os.path.join(mask_dir, f)
                                  for f in os.listdir(mask_dir) if f.lower().endswith('.png')])
        
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(f"The number of images ({len(self.image_paths)}) and masks ({len(self.mask_paths)}) does not match. "
                           f"Images: {[os.path.basename(p) for p in self.image_paths[:3]]}... "
                           f"Masks: {[os.path.basename(p) for p in self.mask_paths[:3]]}...")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read the image.
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise ValueError(f"Image not found: {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Read the mask - try color first to be flexible with formats
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_COLOR)
        if mask is None:
            raise ValueError(f"Mask not found: {self.mask_paths[idx]}")
        
        # Convert to grayscale if needed
        if len(mask.shape) == 3 and mask.shape[2] == 3:
            if np.allclose(mask[:,:,0], mask[:,:,1]) and np.allclose(mask[:,:,1], mask[:,:,2]):
                mask = mask[:,:,0]  # Take first channel
            else:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        elif len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # NORMALIZE MASK VALUES TO 0, 1, 2
        unique_vals = np.unique(mask)
        if len(unique_vals) <= 3:
            sorted_vals = sorted(unique_vals)
            for new_idx, old_val in enumerate(sorted_vals):
                if new_idx >= 3:
                    break
                mask[mask == old_val] = 255 + new_idx
            
            for new_idx, old_val in enumerate(sorted_vals[:3]):
                mask[mask == 255 + new_idx] = new_idx
        else:
            mask_normalized = np.zeros_like(mask, dtype=np.int64)
            mask_normalized[mask < 85] = 0
            mask_normalized[(mask >= 85) & (mask < 170)] = 1
            mask_normalized[mask >= 170] = 2
            mask = mask_normalized
        
        mask = torch.from_numpy(mask.astype(np.int64)).to(torch.int64)
        # Ensure mask has a channel dimension.
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask


# -------------------------------------------------------------------
# Function to save an augmented image and mask.
# The outputs will be saved in "data/augment_01/images" and "data/augment_01/masks".
# -------------------------------------------------------------------
def save_augmented(image: torch.Tensor, mask: torch.Tensor, out_dir_images, out_dir_masks, prefix):
    """Save augmented image and mask to disk."""
    os.makedirs(out_dir_images, exist_ok=True)
    os.makedirs(out_dir_masks, exist_ok=True)

    image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    mask_np = mask.cpu().numpy().astype(np.uint8)

    image_path = os.path.join(out_dir_images, f'{prefix}.png')
    cv2.imwrite(image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    mask_path = os.path.join(out_dir_masks, f'{prefix}.png')
    cv2.imwrite(mask_path, mask_np)

    print(f"Augmented image saved to: {image_path}")
    print(f"Augmented mask saved to: {mask_path}")


def augment_images(
    num_repetitions: int = 1,
    image_dir: str = None,
    mask_dir: str = None,
    output_dir: str = None,
    prefix: str = 'augment',
    config = None
):
    """
    Augment images and masks with random transformations.
    
    Args:
        num_repetitions: Number of times to augment each image
        image_dir: Directory containing images (uses default if None)
        mask_dir: Directory containing masks (uses default if None)
        output_dir: Directory to save augmented images/masks (uses default if None)
        prefix: Prefix for augmented filenames
        config: Optional AugmentationConfig instance (overrides other parameters)
    """
    # Use config if provided
    if config is not None and AugmentationConfig is not None:
        num_repetitions = config.num_repetitions
        prefix = config.prefix
        if config.output_dir:
            output_dir = config.output_dir
    
    # Get project root if directories not specified
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    if image_dir is None:
        image_dir = os.path.join(project_dir, "data", "images")
    if mask_dir is None:
        mask_dir = os.path.join(project_dir, "data", "masks")
    if output_dir is None:
        output_dir = os.path.join(project_dir, "data")
    
    out_dir_images = os.path.join(output_dir, "images")
    out_dir_masks = os.path.join(output_dir, "masks")
    
    # Verify directories exist
    if not os.path.isdir(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")
    if not os.path.isdir(mask_dir):
        raise ValueError(f"Mask directory not found: {mask_dir}")
    
    # Create dataset
    dataset = SegmentationDataset(image_dir, mask_dir, transform=joint_transform)
    print(f"Loaded dataset with {len(dataset)} image-mask pairs")

    # Generate augmented data
    total_generated = 0
    for rep_idx in range(num_repetitions):
        print(f"\nAugmentation pass {rep_idx + 1}/{num_repetitions}")
        for idx in range(len(dataset)):
            image, mask = dataset[idx]
            # Use repetition index and sample index in filename to avoid overwrites
            aug_prefix = f'{prefix}_rep{rep_idx:02d}_sample{idx:02d}'
            save_augmented(image, mask, out_dir_images, out_dir_masks, aug_prefix)
            total_generated += 1
    
    print(f"\nAugmentation complete! Generated {total_generated} augmented image-mask pairs")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Augment image and mask datasets with random transformations"
    )
    parser.add_argument(
        '--repetitions', '-r',
        type=int,
        default=1,
        help='Number of augmentation repetitions per image (default: 1)'
    )
    parser.add_argument(
        '--image-dir',
        help='Directory containing original images (default: data/images)'
    )
    parser.add_argument(
        '--mask-dir',
        help='Directory containing original masks (default: data/masks)'
    )
    parser.add_argument(
        '--output-dir',
        help='Directory to save augmented images/masks (default: data/)'
    )
    parser.add_argument(
        '--prefix',
        default='augment',
        help='Prefix for augmented filenames (default: augment)'
    )
    
    args = parser.parse_args()
    
    try:
        augment_images(
            num_repetitions=args.repetitions,
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            output_dir=args.output_dir,
            prefix=args.prefix
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.__stderr__)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
