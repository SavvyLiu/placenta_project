# Quick Reference Guide - Placenta Segmentation Utils

## Table of Contents
1. [Inference (Model Segmentation)](#inference)
2. [Data Augmentation](#augmentation)
3. [Mask Conversion](#mask-conversion)
4. [Mask Cleanup](#mask-cleanup)

---

## Inference

### Segment a Single Image
```bash
python -m utilities.inference \
  --input data/images/sample.png \
  --output-mask results/mask.png \
  --output-annot results/annotated.png
```

### Specify Architecture
```bash
# EfficientNet (default)
python -m utilities.inference --arch efficientnet --input image.png \
  --output-mask mask.png --output-annot annotated.png

# ResNet34 U-Net
python -m utilities.inference --arch unet --input image.png \
  --output-mask mask.png --output-annot annotated.png

# RegNet
python -m utilities.inference --arch regnet --input image.png \
  --output-mask mask.png --output-annot annotated.png

# Vision Transformer
python -m utilities.inference --arch vit --input image.png \
  --output-mask mask.png --output-annot annotated.png
```

### Set Custom Model Path
```bash
python -m utilities.inference \
  --arch efficientnet \
  --model-path path/to/custom_model.pth \
  --input image.png \
  --output-mask mask.png
```

### Evaluate with Ground Truth
```bash
python -m utilities.inference \
  --input data/validation/test.png \
  --output-mask results/test_mask.png \
  --output-annot results/test_annot.png \
  --ground-truth data/validation/test_gt.png \
  --compute-metrics
```

### Adjust Post-Processing
```bash
# No morphological operations
python -m utilities.inference \
  --input image.png \
  --output-mask mask.png \
  --morphology none

# Closing instead of opening
python -m utilities.inference \
  --input image.png \
  --output-mask mask.png \
  --morphology close

# Custom kernel size
python -m utilities.inference \
  --input image.png \
  --output-mask mask.png \
  --kernel-size 5
```

### Filter Small Contours
```bash
# Only show contours with area > 100 pixels
python -m utilities.inference \
  --input image.png \
  --output-annot annotated.png \
  --min-contour-area 100
```

### Use CPU Only
```bash
python -m utilities.inference \
  --input image.png \
  --output-mask mask.png \
  --device cpu
```

---

## Data Augmentation

### Augment Training Set (3 repetitions)
```bash
python -m utilities.image_augmentation \
  --repetitions 3
```

### Custom Output Directory
```bash
python -m utilities.image_augmentation \
  --repetitions 2 \
  --output-dir data/augmented_v2
```

### Custom Input/Output Paths
```bash
python -m utilities.image_augmentation \
  --repetitions 3 \
  --image-dir data/raw_images \
  --mask-dir data/raw_masks \
  --output-dir data/augmented
```

### Custom Prefix for Generated Files
```bash
python -m utilities.image_augmentation \
  --repetitions 2 \
  --prefix my_augmented
```

**Generated filenames:**
```
my_augmented_rep00_sample00.png
my_augmented_rep00_sample01.png
my_augmented_rep01_sample00.png
...
```

---

## Mask Conversion

### Convert Single Mask (Threshold Method)
```bash
python -m utilities.colour_convert \
  --input data/masks/raw.png \
  --output data/masks/binary.png \
  --method threshold \
  --threshold 127
```

### Detect Grey Regions
```bash
python -m utilities.colour_convert \
  --input colored_mask.png \
  --output binary_mask.png \
  --method grey
```

### Batch Convert Directory
```bash
python -m utilities.colour_convert \
  --input data/masks_raw/ \
  --output data/masks_clean/ \
  --method threshold \
  --batch
```

### Different Threshold Value
```bash
# Lower threshold (be more inclusive)
python -m utilities.colour_convert \
  --input mask.png \
  --output binary.png \
  --threshold 100

# Higher threshold (be more conservative)
python -m utilities.colour_convert \
  --input mask.png \
  --output binary.png \
  --threshold 150
```

---

## Mask Cleanup

### Clean Masks (Convert Non-Black to White)
```bash
python -m utilities.white --dir data/masks
```

### Create Backups Before Processing
```bash
python -m utilities.white --dir data/masks --backup
```

Creates `mask.png.backup` for each file.

### Preview Changes (Dry-Run)
```bash
python -m utilities.white --dir data/masks --dry-run
```

Shows what would be processed without making changes.

### Custom File Pattern
```bash
# Process only files starting with "mask_"
python -m utilities.white --dir data/masks --pattern "mask_*.png"
```

---

## Output Files Explanation

### Segmentation Mask (`--output-mask`)
- **Format:** Single-channel PNG with values:
  - `0` = Background (black)
  - `1` = Fetal tissue
  - `2` = Maternal tissue
- **Usage:** Direct quantitative analysis, metrics computation

### Annotated Image (`--output-annot`)
- **Format:** RGB PNG with color-coded overlays
  - Background: Black (no color)
  - Fetal: Green contours
  - Maternal: Red contours
- **Usage:** Visual inspection, presentation, reports

### Metrics
When `--ground-truth` and `--compute-metrics` are provided:
- **Per-class:** Fetal IoU, Maternal IoU, Fetal Dice, Maternal Dice
- **Averaged:** Mean IoU, Mean Dice
- **Healthcare:** Useful for model comparison and validation

---

## Common Workflows

### Complete Training Pipeline
```bash
# 1. Augment training data
python -m utilities.image_augmentation --repetitions 3

# 2. Train models
python -m models.train_all_models \
  --models efficientnet regnet unet \
  --epochs 100

# 3. Evaluate on validation set
python -m utilities.inference \
  --arch efficientnet \
  --input data/validation/test1.png \
  --output-mask results/test1_mask.png \
  --output-annot results/test1_annot.png \
  --ground-truth data/validation/test1_gt.png \
  --compute-metrics
```

### Batch Inference on New Data
```bash
# Process validation set with best model
for img in data/validation/images/*.png; do
  base=$(basename "$img" .png)
  python -m utilities.inference \
    --arch efficientnet \
    --input "$img" \
    --output-mask "results/${base}_mask.png" \
    --output-annot "results/${base}_annot.png"
done
```

### Data Cleaning & Augmentation
```bash
# 1. Clean raw masks
python -m utilities.white --dir data/masks_raw --backup

# 2. Convert format if needed
python -m utilities.colour_convert \
  --input data/masks_raw/ \
  --output data/masks/ \
  --method threshold \
  --batch

# 3. Augment for training
python -m utilities.image_augmentation --repetitions 5
```

---

## Troubleshooting

### "Model file not found"
- Check that trained models are in `trained_models/` directory
- Or specify explicit path with `--model-path`

### "No module named 'utilities'"
- Run commands from project root directory
- Ensure you're using `python -m utilities.inference` (not just `inference.py`)

### "CUDA out of memory"
- Use `--device cpu` to run on CPU instead
- Or reduce image resolution before inference

### "No ground truth found"
- Ground truth is optional; just don't use `--ground-truth` flag
- If provided, must match prediction mask size

### Output mask looks all black
- Increase logging: add `--verbose` flag
- Check that model file exists and loaded correctly
- Verify input image is valid

---

## Tips & Best Practices

1. **Always backup data before batch processing:**
   ```bash
   python -m utilities.white --dir data/masks --backup
   ```

2. **Use dry-run before batch operations:**
   ```bash
   python -m utilities.white --dir data/masks --dry-run
   ```

3. **Save annotated images for visual inspection:**
   ```bash
   python -m utilities.inference \
     --input image.png \
     --output-annot results/visual_check.png
   ```

4. **Evaluate on validation set regularly:**
   ```bash
   python -m utilities.inference \
     --input validation/test.png \
     --ground-truth validation/test_gt.png \
     --compute-metrics
   ```

5. **Use consistent morphological operations:**
   ```bash
   # Same kernel size for all images
   python -m utilities.inference \
     --input image.png \
     --morphology open \
     --kernel-size 5
   ```

---

## Configuration via Python

For programmatic usage:

```python
from utilities.config import InferenceConfig
from utilities.model_loader import load_model
from utilities.inference import infer_single_image
import torch

# Create configuration
config = InferenceConfig(
    architecture='efficientnet',
    input_image='data/test.png',
    output_mask='results/mask.png',
    output_annotated='results/annotated.png',
    device='auto',
    threshold=0.5,
    morphology_operation='open',
    morphology_kernel_size=(3, 3),
    min_contour_area=10,
    ground_truth_path='data/gt.png',
    compute_metrics=True
)

# Run inference
results = infer_single_image(config)

# Access results
mask = results['mask']  # numpy array with class indices
metrics = results['metrics']  # dict with IoU, Dice scores
areas = results['areas']  # dict with pixel areas per class
```

---

## Version Notes

- All changes support **3-class segmentation** (Background, Fetal, Maternal)
- All utilities have **CLI support** (run with `--help` for details)
- Requires **PyTorch**, **OpenCV**, **NumPy** (see main `requirements.txt`)
- Tested on **CUDA 12.x** and **CPU** implementations

