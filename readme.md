# Placenta Project

## Purpose

The purpose of this project is to apply machine learning techniques to automate the segmentation of placental histology slides. 

## Quick Start

### A) Model Training

1. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```
2. (Optional) Use a small subset of 4 images for debugging:
   ```cmd
   python -m models.train_all_models --models unet --subset
   ```
3. Train one or more models:
   ```cmd
   # Train a single model (ResNet34 U-Net):
   python -m models.train_UNET_smp

   # Train multiple backbones:
   python -m models.train_all_models --models efficientnet regnet vit unet --epochs 50
   ```
4. Checkpoints are saved in `trained_models/`:
   ```text
   trained_models/
     smp_unet_placenta.pth
     regnet_unet_placenta.pth
     efficientnet_unet_placenta.pth
     vit_unet_placenta_flexible.pth
   ```

### B) Inference & Evaluation

1. Run segmentation on a new image:
   ```cmd
   python -m utilities.inference \
     --arch regnet \
     --input data/validation/01.png \
     --output_mask results/01_mask.png \
     --output_annot results/01_annot.png
   ```
   - If `--model_path` is omitted, the script uses the corresponding file in `trained_models/`.
2. The script saves:
   - A binary mask at `--output_mask`.
   - An annotated image with contours at `--output_annot`.
   - Prints IoU and Dice scores against `data/validation/ground_truth/`.

### C) Utilities

- **Mask conversion**: Convert grey-scale masks to binary:
  ```cmd
  python -c "from utilities.colour_convert import convert_grey_foreground_mask; \
  convert_grey_foreground_mask('data/masks/16.png','bin_16.png')"
  ```

- **Augmentation**: Generate augmented images and masks:
  ```cmd
  python -m utilities.image_augmentation
  ```
  Saves `augment_XX.png` files to `data/images/` and `data/masks/`.

## Project Structure

```
<project root>
|-- data/
|-- models/
|-- trained_models/
|-- utilities/
|-- readme.md
|-- requirements.txt
```

## Contact

If you have any questions or feedback, please feel free to reach out to the project maintainers.
