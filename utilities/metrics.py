"""
Validation metrics for semantic segmentation.
Computes per-class IoU (Intersection over Union), Dice coefficient, and overall accuracy.
"""

import torch
import torch.nn as nn


class SegmentationMetrics:
    """Compute segmentation metrics: IoU, Dice, Accuracy per class."""
    
    def __init__(self, num_classes=3, class_names=None, ignore_index=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        if class_names is None:
            self.class_names = {i: f'class_{i}' for i in range(num_classes)}
        else:
            self.class_names = class_names
    
    def compute_iou(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Compute per-class IoU (Intersection over Union).
        
        Args:
            pred: Predicted class indices, shape (B, H, W) or (B, C, H, W) after argmax
            target: Ground truth class indices, shape (B, H, W)
        
        Returns:
            Dictionary with per-class IoU and mean IoU
        """
        # If pred has channel dimension, take argmax
        if pred.dim() == 4:
            pred = torch.argmax(pred, dim=1)
        
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        
        iou_scores = {}
        valid_classes = []
        
        for class_idx in range(self.num_classes):
            pred_mask = (pred == class_idx)
            target_mask = (target == class_idx)
            
            intersection = (pred_mask & target_mask).sum()
            union = (pred_mask | target_mask).sum()
            
            if union == 0:
                iou = 1.0 if intersection == 0 else 0.0
            else:
                iou = intersection / union
            
            class_name = self.class_names.get(class_idx, f'class_{class_idx}')
            iou_scores[class_name] = float(iou)
            valid_classes.append(iou)
        
        iou_scores['mean_iou'] = sum(valid_classes) / len(valid_classes) if valid_classes else 0.0
        return iou_scores
    
    def compute_dice(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Compute per-class Dice coefficient (F1 score).
        
        Args:
            pred: Predicted class indices, shape (B, H, W) or (B, C, H, W) after argmax
            target: Ground truth class indices, shape (B, H, W)
        
        Returns:
            Dictionary with per-class Dice and mean Dice
        """
        # If pred has channel dimension, take argmax
        if pred.dim() == 4:
            pred = torch.argmax(pred, dim=1)
        
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        
        dice_scores = {}
        valid_classes = []
        
        for class_idx in range(self.num_classes):
            pred_mask = (pred == class_idx).astype(float)
            target_mask = (target == class_idx).astype(float)
            
            intersection = (pred_mask * target_mask).sum()
            union = pred_mask.sum() + target_mask.sum()
            
            if union == 0:
                dice = 1.0 if intersection == 0 else 0.0
            else:
                dice = (2.0 * intersection) / union
            
            class_name = self.class_names.get(class_idx, f'class_{class_idx}')
            dice_scores[class_name] = float(dice)
            valid_classes.append(dice)
        
        dice_scores['mean_dice'] = sum(valid_classes) / len(valid_classes) if valid_classes else 0.0
        return dice_scores
    
    def compute_accuracy(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Compute per-class and overall accuracy.
        
        Args:
            pred: Predicted class indices, shape (B, H, W) or (B, C, H, W) after argmax
            target: Ground truth class indices, shape (B, H, W)
        
        Returns:
            Dictionary with per-class accuracy and overall accuracy
        """
        # If pred has channel dimension, take argmax
        if pred.dim() == 4:
            pred = torch.argmax(pred, dim=1)
        
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        
        acc_scores = {}
        overall_correct = (pred == target).sum()
        overall_total = target.size
        
        for class_idx in range(self.num_classes):
            target_mask = (target == class_idx)
            if target_mask.sum() == 0:
                acc = 0.0
            else:
                correct = ((pred == class_idx) & target_mask).sum()
                acc = correct / target_mask.sum()
            
            class_name = self.class_names.get(class_idx, f'class_{class_idx}')
            acc_scores[class_name] = float(acc)
        
        acc_scores['overall_accuracy'] = float(overall_correct / overall_total)
        return acc_scores
    
    def compute_all(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Compute all metrics at once.
        
        Args:
            pred: Predicted class indices or logits
            target: Ground truth class indices
        
        Returns:
            Dictionary containing IoU, Dice, and Accuracy metrics
        """
        metrics = {
            'iou': self.compute_iou(pred, target),
            'dice': self.compute_dice(pred, target),
            'accuracy': self.compute_accuracy(pred, target)
        }
        return metrics
    
    def format_metrics(self, metrics):
        """Format metrics dictionary for logging."""
        lines = []
        lines.append("=== Segmentation Metrics ===")
        
        if 'iou' in metrics:
            lines.append("IoU:")
            for k, v in metrics['iou'].items():
                lines.append(f"  {k}: {v:.4f}")
        
        if 'dice' in metrics:
            lines.append("Dice:")
            for k, v in metrics['dice'].items():
                lines.append(f"  {k}: {v:.4f}")
        
        if 'accuracy' in metrics:
            lines.append("Accuracy:")
            for k, v in metrics['accuracy'].items():
                lines.append(f"  {k}: {v:.4f}")
        
        return "\n".join(lines)
