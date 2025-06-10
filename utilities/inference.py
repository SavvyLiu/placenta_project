import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import os
import argparse
from models.efficicentnet_train_smp import EfficientNetUNet
from models.regnet_train_smp import RegNetUNet
from models.ViT_train_smp import ViT_UNet_Flexible


def segment_image_smp(arch, model_path, input_image_path, output_mask_path):
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

    # Initialize the model based on chosen architecture
    if arch == 'unet':
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )
    elif arch == 'efficientnet':
        model = EfficientNetUNet(n_classes=1)
    elif arch == 'regnet':
        model = RegNetUNet(n_classes=1)
    elif arch == 'vit':
        model = ViT_UNet_Flexible(n_classes=1)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded successfully")
    model.to(device)
    model.eval()

    # Load and preprocess the image
    img = cv2.imread(input_image_path)
    if img is None:
        raise ValueError(f"Could not load image at {input_image_path}")
    print(f"Loaded image shape: {img.shape}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    print(f"Image value range: [{img_rgb.min():.3f}, {img_rgb.max():.3f}]")
    tensor_img = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    print(f"Input tensor shape: {tensor_img.shape}")

    # Inference
    with torch.no_grad():
        output = model(tensor_img)  # shape: (1, 1, H, W)
        print(f"Raw model output shape: {output.shape}")
        print(f"Raw output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        prob_map = torch.sigmoid(output)
        print(f"After sigmoid range: [{prob_map.min().item():.3f}, {prob_map.max().item():.3f}]")
        print(f"Pixels > 0.5: {(prob_map > 0.5).sum().item()}")
        mask_pred = (prob_map > 0.5).float().cpu().numpy()[0, 0]
        print(f"Final mask has {mask_pred.sum()} white pixels")

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


def load_mask(image_path):
    """
    Loads an image as a grayscale mask and thresholds it to obtain a binary mask.
    Assumes the image is saved such that pixels >127 represent the segmented area.
    """
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    # Convert grayscale mask to binary: pixels above 50 are set to 1, else 0.
    _, binary_mask = cv2.threshold(mask, 50, 1, cv2.THRESH_BINARY)
    return binary_mask

def iou_score(y_true, y_pred):
    

    """
    Computes the Intersection over Union (IoU) score.
    """
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    if np.sum(union) == 0:
        return 1.0  # Both masks are empty.
    return np.sum(intersection) / np.sum(union)


def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    dice = (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))
    return dice



def main():
    parser = argparse.ArgumentParser(description="Segment an image with a chosen placenta model architecture")
    parser.add_argument("--arch", choices=["unet","efficientnet","regnet","vit"], default="unet")
    parser.add_argument("--model_path", help="Path to .pth weights file; if omitted, will use trained_models/<arch>*.pth")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output_mask", required=True, help="Path to save the binary mask")
    parser.add_argument("--output_annot", required=True, help="Path to save annotated image with contours")
    args = parser.parse_args()

    # Infer default model path if not provided
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    if args.model_path:
        model_path = args.model_path
    else:
        # map arch to default filename
        names = {
            'unet': 'trained_models/smp_unet_placenta.pth',
            'efficientnet': 'trained_models/efficientnet_unet_placenta.pth',
            'regnet': 'trained_models/regnet_unet_placenta.pth',
            'vit': 'trained_models/vit_unet_placenta_flexible.pth'
        }
        model_path = os.path.join(project_dir, names[args.arch])
    # Construct ground-truth filename by prefixing 'valid' to the base name
    base = os.path.splitext(os.path.basename(args.input))[0]
    gt_filename = f"valid{base}.png"
    ground_truth_path = os.path.join(project_dir, 'data', 'validation', 'ground_truth', gt_filename)

    # Step 1: Segment
    mask = segment_image_smp(args.arch, model_path, args.input, args.output_mask)

    # Step 2: Draw contours and evaluate
    draw_contours_on_masked_image(args.input, mask, args.output_annot)
    accuracy(args.output_mask, ground_truth_path)
    close_open(args.output_mask, ground_truth_path)
 
def accuracy(output_mask_path, ground_truth_path):

    y_true = load_mask(ground_truth_path)
    y_pred = load_mask(output_mask_path)
    print(iou_score(y_true, y_pred))
    print(dice_coefficient(y_true, y_pred))

def close_open(output_mask_path, ground_truth_path):
    gt_bin = load_mask(ground_truth_path)
    pred_bin = load_mask(output_mask_path)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    # Erosion: shrinks foreground
    pred_eroded = cv2.erode(pred_bin, kernel)

    # Dilation: expands foreground
    pred_dilated = cv2.dilate(pred_bin, kernel)

    # Opening: erosion followed by dilation (removes small noise)
    pred_opened = cv2.morphologyEx(pred_bin, cv2.MORPH_OPEN, kernel)

    # Closing: dilation followed by erosion (fills small holes)
    pred_closed = cv2.morphologyEx(pred_bin, cv2.MORPH_CLOSE, kernel)

    # 5. Compute IoU after each operation
    iou_eroded = iou_score(gt_bin, pred_eroded)
    iou_dilated = iou_score(gt_bin, pred_dilated)
    iou_opened = iou_score(gt_bin, pred_opened)
    iou_closed = iou_score(gt_bin, pred_closed)

    print("Eroded IoU:", iou_eroded)
    print("Dilated IoU:", iou_dilated)
    print("Opened IoU:", iou_opened)
    print("Closed IoU:", iou_closed)
    cv2.imwrite('pred_eroded.png', pred_eroded * 255)
    cv2.imwrite('pred_dilated.png', pred_dilated * 255)
    cv2.imwrite('pred_opened.png', pred_opened * 255)
    cv2.imwrite('pred_closed.png', pred_closed * 255)
    

if __name__ == "__main__":
    main()
