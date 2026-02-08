"""
Unified Inference Pipeline for Placenta Segmentation
Supports: U-Net, EfficientNet, RegNet, Vision Transformer
Handles: 3-class segmentation (Background, Fetal, Maternal)
"""

import os
import sys
import cv2
import torch
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List

from utilities.model_loader import load_model, get_device, CLASS_LABELS, CLASS_COLORS
from utilities.config import InferenceConfig, get_device as config_get_device


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def segment_image(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Run inference on a single image and generate class segmentation.
    
    Args:
        model: Loaded model in eval mode
        image_path: Path to input image
        device: Device to run inference on
        threshold: Classification threshold (for binary decisions if needed)
    
    Returns:
        Segmentation mask with class indices (0, 1, 2)
    """
    # Load and preprocess image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    logger.info(f"Loaded image shape: {img_bgr.shape}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    
    # Convert to tensor
    tensor_img = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    logger.info(f"Input tensor shape: {tensor_img.shape}")
    
    # Run inference
    with torch.no_grad():
        logits = model(tensor_img)  # shape: (1, 3, H, W)
        logger.info(f"Model output shape: {logits.shape}")
        logger.info(f"Output range: [{logits.min():.3f}, {logits.max():.3f}]")
        
        # Get class predictions via argmax
        prob_map = torch.softmax(logits, dim=1)  # (1, 3, H, W)
        pred_classes = torch.argmax(prob_map, dim=1)[0].cpu().numpy()  # (H, W)
        logger.info(f"Predicted classes: {np.unique(pred_classes)}")
        logger.info(f"Class distribution: {np.bincount(pred_classes.flatten())}")
    
    return pred_classes.astype(np.uint8)


def apply_morphological_operations(
    mask: np.ndarray,
    operation: str = 'open',
    kernel_size: Tuple[int, int] = (3, 3)
) -> np.ndarray:
    """
    Apply morphological operations to refine the segmentation mask.
    
    Args:
        mask: Input mask with class indices
        operation: 'open', 'close', 'both', or 'none'
        kernel_size: Size of morphological kernel
    
    Returns:
        Refined mask
    """
    if operation == 'none':
        return mask
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    refined_mask = mask.copy()
    
    # Apply operations only on non-background regions
    for class_id in [1, 2]:  # Fetal and Maternal
        class_mask = (mask == class_id).astype(np.uint8) * 255
        
        if operation in ['open', 'both']:
            class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
        if operation in ['close', 'both']:
            class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)
        
        refined_mask[class_mask > 0] = class_id
        refined_mask[class_mask == 0] = 0
    
    return refined_mask


def save_segmentation_mask(mask: np.ndarray, output_path: str) -> None:
    """Save segmentation mask as PNG."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, mask)
    logger.info(f"Segmentation mask saved to {output_path}")


def create_color_visualization(
    original_image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create a color-coded visualization of the segmentation.
    
    Args:
        original_image: Original image in BGR format
        mask: Segmentation mask with class indices (0, 1, 2)
        alpha: Blending factor (0=original, 1=mask)
    
    Returns:
        Blended image with color-coded classes
    """
    color_mask = np.zeros_like(original_image)
    
    for class_id, color in CLASS_COLORS.items():
        class_pixels = mask == class_id
        color_mask[class_pixels] = color
    
    # Blend with original image
    blended = cv2.addWeighted(original_image, 1 - alpha, color_mask, alpha, 0)
    return blended


def draw_contours_on_image(
    original_image: np.ndarray,
    mask: np.ndarray,
    min_area: int = 10,
    use_bounding_box: bool = False,
    contour_color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Find contours in the mask and draw them on the original image.
    
    Args:
        original_image: Original image in BGR format
        mask: Segmentation mask with class indices
        min_area: Minimum contour area to include
        use_bounding_box: If True, draw bounding boxes instead of contours
        contour_color: RGB color for contours (BGR for OpenCV)
        thickness: Line thickness for contours
    
    Returns:
        Tuple of (annotated_image, area_dict)
        area_dict contains total area for each class
    """
    annotated = original_image.copy()
    area_dict = {0: 0, 1: 0, 2: 0}
    
    # Process each class separately
    for class_id in [1, 2]:  # Skip background
        class_mask = (mask == class_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        class_color = CLASS_COLORS[class_id]
        total_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            total_area += area
            
            if use_bounding_box:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), class_color, thickness)
            else:
                # Draw smooth contour
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                cv2.drawContours(annotated, [approx], -1, class_color, thickness)
        
        area_dict[class_id] = total_area
    
    return annotated, area_dict


def compute_metrics(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray
) -> Dict[str, float]:
    """
    Compute segmentation metrics (IoU, Dice) for each class.
    
    Args:
        predicted_mask: Predicted segmentation (class indices)
        ground_truth_mask: Ground truth segmentation
    
    Returns:
        Dictionary of metrics per class
    """
    metrics = {}
    
    for class_id in range(3):
        pred_binary = (predicted_mask == class_id).astype(np.float32)
        gt_binary = (ground_truth_mask == class_id).astype(np.float32)
        
        # IoU
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        iou = intersection / union if union > 0 else 1.0
        
        # Dice
        dice = (2.0 * intersection) / (pred_binary.sum() + gt_binary.sum()) if (pred_binary.sum() + gt_binary.sum()) > 0 else 1.0
        
        class_name = CLASS_LABELS[class_id]
        metrics[f'{class_name}_iou'] = float(iou)
        metrics[f'{class_name}_dice'] = float(dice)
    
    # Macro-averaged metrics
    iou_values = [metrics[f'{CLASS_LABELS[i]}_iou'] for i in range(1, 3)]  # Exclude background
    dice_values = [metrics[f'{CLASS_LABELS[i]}_dice'] for i in range(1, 3)]
    
    metrics['mean_iou'] = float(np.mean(iou_values))
    metrics['mean_dice'] = float(np.mean(dice_values))
    
    return metrics


def load_ground_truth(gt_path: str) -> Optional[np.ndarray]:
    """Load ground truth mask from file."""
    if not os.path.exists(gt_path):
        logger.warning(f"Ground truth not found at {gt_path}")
        return None
    
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        logger.warning(f"Could not load ground truth mask at {gt_path}")
        return None
    
    return gt_mask.astype(np.uint8)


def infer_single_image(
    config: InferenceConfig
) -> Dict:
    """
    Run inference on a single image with the specified configuration.
    
    Args:
        config: InferenceConfig object
    
    Returns:
        Dictionary with results and metrics
    """
    # Setup device
    device = config_get_device(config.device)
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading {config.architecture} model...")
    model = load_model(
        architecture=config.architecture,
        model_path=config.model_path,
        device=device,
        n_classes=3
    )
    
    # Segment image
    logger.info(f"Segmenting image: {config.input_image}")
    pred_mask = segment_image(model, config.input_image, device, config.threshold)
    
    # Apply morphological operations
    if config.morphology_operation != 'none':
        logger.info(f"Applying morphological {config.morphology_operation}...")
        pred_mask = apply_morphological_operations(
            pred_mask,
            operation=config.morphology_operation,
            kernel_size=config.morphology_kernel_size
        )
    
    # Save segmentation mask
    if config.output_mask:
        save_segmentation_mask(pred_mask, config.output_mask)
    
    # Create and save annotated image
    results = {
        'mask': pred_mask,
        'metrics': {}
    }
    
    if config.output_annotated:
        original_img = cv2.imread(config.input_image)
        annotated_img, area_dict = draw_contours_on_image(
            original_img,
            pred_mask,
            min_area=config.min_contour_area,
            use_bounding_box=config.use_bounding_box,
            contour_color=config.contour_color,
            thickness=config.contour_thickness
        )
        
        os.makedirs(os.path.dirname(config.output_annotated), exist_ok=True)
        cv2.imwrite(config.output_annotated, annotated_img)
        logger.info(f"Annotated image saved to {config.output_annotated}")
        
        results['areas'] = area_dict
        logger.info(f"Segmented areas - Fetal: {area_dict[1]:.0f} px, Maternal: {area_dict[2]:.0f} px")
    
    # Compute metrics if ground truth available
    if config.compute_metrics and config.ground_truth_path:
        gt_mask = load_ground_truth(config.ground_truth_path)
        if gt_mask is not None:
            metrics = compute_metrics(pred_mask, gt_mask)
            results['metrics'] = metrics
            
            logger.info("=== Segmentation Metrics ===")
            for class_id in range(1, 3):
                class_name = CLASS_LABELS[class_id]
                iou = metrics[f'{class_name}_iou']
                dice = metrics[f'{class_name}_dice']
                logger.info(f"{class_name.capitalize()}: IoU={iou:.4f}, Dice={dice:.4f}")
            
            logger.info(f"Mean IoU: {metrics['mean_iou']:.4f}")
            logger.info(f"Mean Dice: {metrics['mean_dice']:.4f}")
    
    return results


def main():
    """Command-line interface for inference."""
    parser = argparse.ArgumentParser(
        description="Placenta Segmentation Inference Pipeline"
    )
    
    # Model arguments
    parser.add_argument(
        '--arch', '--architecture',
        choices=['unet', 'efficientnet', 'regnet', 'vit'],
        default='efficientnet',
        help='Model architecture (default: efficientnet)'
    )
    parser.add_argument(
        '--model-path',
        help='Path to model weights. If omitted, uses trained_models/<arch>*.pth'
    )
    
    # Input/output arguments
    parser.add_argument(
        '--input', required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--output-mask',
        help='Path to save predicted mask'
    )
    parser.add_argument(
        '--output-annot',
        help='Path to save annotated image with contours'
    )
    
    # Inference parameters
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Classification threshold (default: 0.5)'
    )
    parser.add_argument(
        '--device', choices=['auto', 'cuda', 'cpu'], default='auto',
        help='Device to use (auto/cuda/cpu, default: auto)'
    )
    parser.add_argument(
        '--morphology', choices=['open', 'close', 'both', 'none'], default='open',
        help='Morphological operation to apply (default: open)'
    )
    parser.add_argument(
        '--kernel-size', type=int, default=3,
        help='Morphological kernel size (default: 3)'
    )
    parser.add_argument(
        '--min-contour-area', type=int, default=10,
        help='Minimum contour area in pixels (default: 10)'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--ground-truth',
        help='Path to ground truth mask for evaluation'
    )
    parser.add_argument(
        '--compute-metrics', action='store_true',
        help='Compute evaluation metrics if ground truth is provided'
    )
    
    # Logging
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = InferenceConfig(
        architecture=args.arch,
        model_path=args.model_path,
        input_image=args.input,
        output_mask=args.output_mask,
        output_annotated=args.output_annot,
        device=args.device,
        threshold=args.threshold,
        morphology_operation=args.morphology,
        morphology_kernel_size=(args.kernel_size, args.kernel_size),
        min_contour_area=args.min_contour_area,
        ground_truth_path=args.ground_truth,
        compute_metrics=args.compute_metrics or args.ground_truth is not None,
        verbose=args.verbose
    )
    
    # Run inference
    try:
        results = infer_single_image(config)
        logger.info("Inference completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
