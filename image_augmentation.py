import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision.transforms.v2 import ColorJitter, InterpolationMode
import torchvision.transforms.functional as TF


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
    def __init__(self, affine_params=None, color_jitter_params=None):
        if affine_params is None:
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
        # List only .png files (adjust if needed)
        self.image_paths = sorted([os.path.join(image_dir, f)
                                   for f in os.listdir(image_dir) if f.lower().endswith('.png')])
        self.mask_paths = sorted([os.path.join(mask_dir, f)
                                  for f in os.listdir(mask_dir) if f.lower().endswith('.png')])
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("The number of images and masks does not match.")
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

        # Read the mask.
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask not found: {self.mask_paths[idx]}")
        mask = torch.from_numpy(mask).to(torch.int64)
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


if __name__ == '__main__':
    # Define input directories.
    image_dir = 'data/images'
    mask_dir = 'data/masks'

    # Create dataset.
    dataset = SegmentationDataset(image_dir, mask_dir, transform=joint_transform)

    # Output directories for augmented results.
    out_dir_images = 'data/images'
    out_dir_masks = 'data/masks'

    # Process every item in the dataset.
    for idx in range(len(dataset)):
        image, mask = dataset[idx]
        prefix = f'augment_{idx:02d}'
        save_augmented(image, mask, out_dir_images, out_dir_masks, prefix)
