import cv2
import numpy as np

def convert_grey_foreground_mask(mask_path, output_binary_path):
    """
    Converts a colored mask with a grey foreground and a black background
    into a binary mask where the grey regions are white (255) and the background is black (0).

    Parameters:
      - mask_path: Path to the input colored mask image.
      - output_binary_path: Path to save the resulting binary mask.

    Returns:
      - binary_mask: The binary mask as a numpy array.
    """
    # Read the mask image (BGR format)
    mask = cv2.imread(mask_path)

    # Convert to HSV color space
    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    # Define thresholds for grey (low saturation and mid-range brightness)
    lower_grey = np.array([0, 0, 50])  # Low saturation, mid brightness
    upper_grey = np.array([180, 50, 200])  # Allow slight color variations but keep low saturation

    # Create a mask for grey regions
    grey_mask = cv2.inRange(hsv, lower_grey, upper_grey)

    cv2.imwrite(output_binary_path, grey_mask)
    print(f"Binary mask saved to {output_binary_path}")
    return grey_mask

# Example usage:
binary_mask = convert_grey_foreground_mask("data/masks/07.png", "data/masks/07.png")
