import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp


def segment_image_smp(model_path, input_image_path, output_mask_path):
    """
    Loads a pretrained segmentation model and uses it to segment the input image.
    The predicted mask is refined with a morphological opening operation.

    Parameters:
      - model_path: Path to the saved model weights.
      - input_image_path: Path to the input histology image.
      - output_mask_path: Path where the binary mask will be saved.

    Returns:
      - refined_mask: The refined binary mask as a numpy array.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model (U-Net with ResNet34 backbone)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and preprocess the image
    img = cv2.imread(input_image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    tensor_img = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(tensor_img)  # shape: (1, 1, H, W)
        prob_map = torch.sigmoid(output)
        mask_pred = (prob_map > 0.5).float().cpu().numpy()[0, 0]

    mask_pred_uint8 = (mask_pred * 255).astype(np.uint8)

    # Refine the mask using a morphological opening (to remove small noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined_mask = cv2.morphologyEx(mask_pred_uint8, cv2.MORPH_OPEN, kernel)

    cv2.imwrite(output_mask_path, refined_mask)
    print(f"Segmentation mask saved to {output_mask_path}")
    return refined_mask


def draw_contours_on_masked_image(input_image_path, mask, output_annotated_path, min_area=10, use_bounding_box=False):
    """
    Finds contours in the provided binary mask and draws either fluid contours (following the actual shape)
    or bounding boxes on the original image. Also computes the total area covered by these contours.

    Parameters:
      - input_image_path: Path to the original histology image.
      - mask: The binary mask (numpy array) where contours are detected.
      - output_annotated_path: Path to save the annotated image.
      - min_area: Minimum contour area to filter out small detections.
      - use_bounding_box: If True, draw bounding boxes instead of the actual contour outlines.
    """
    # Load the original image for annotation
    img = cv2.imread(input_image_path)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = img.copy()

    total_area = 0  # To accumulate area of valid contours

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter out very small contours that might be noise
        if area < min_area:
            continue

        total_area += area

        if use_bounding_box:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle
        else:
            # Draw the fluid contour directly; this follows the shape of the detected area.
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(annotated, [approx], -1, (0, 255, 0), 2)  # Green contour

    # Save the annotated image
    cv2.imwrite(output_annotated_path, annotated)
    print(f"Annotated image saved to {output_annotated_path}")

    # Print the total area of all valid contours
    print(f"Total segmented area (in pixels): {total_area:.2f}")


def main():
    # Define file paths
    model_path = "smp_unet_placenta.pth"           # Trained model weights
    input_image_path = "data/validation/01.png"      # Input histological image
    output_mask_path = "test_mask_pred.png"          # Where to save the segmentation mask
    output_annotated_path = "test_image_annotated.jpg"  # Where to save the annotated image

    # Step 1: Segment the image and save the binary mask
    mask = segment_image_smp(model_path, input_image_path, output_mask_path)

    # Step 2: Draw contours on the image based on the mask.
    # Set use_bounding_box=True if you prefer bounding boxes.
    draw_contours_on_masked_image(input_image_path, mask, output_annotated_path, min_area=10, use_bounding_box=False)


if __name__ == "__main__":
    main()
