#!/usr/bin/env python3
import os
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision

# -------------------------------------
# 1. Configuration (edit only if paths change)
# -------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# Validation images and ground-truth masks
IMAGES_DIR  = os.path.join(PROJECT_DIR, "data", "validation")
MASKS_DIR   = os.path.join(PROJECT_DIR, "data", "validation", "ground_truth")

# Output folder for predictions
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "data", "validation", "predictions")

# Model checkpoint (saved during training)
CHECKPOINT  = os.path.join(SCRIPT_DIR, "efficientnet_unet_placenta.pth")

# Inference parameters
BATCH_SIZE  = 1
THRESHOLD   = 0.5

# -------------------------------------
# 2. Model definition (matching training)
# -------------------------------------
class EfficientNetUNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        # 1) keep `self.encoder` so checkpoints match
        weights      = torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
        self.encoder = torchvision.models.efficientnet_v2_l(weights=weights)
        # 2) alias its features for forward
        self.encoder_features = self.encoder.features

        # Decoder upsampling blocks
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, 2, 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 2, 2),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, 2),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  2, 2),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32,  2, 2),  nn.ReLU(inplace=True),
        )
        # Final 1x1 conv for binary mask
        self.final_conv = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        features = self.encoder_features(x)
        x = self.decoder(features)
        return self.final_conv(x)

# -------------------------------------
# 3. Inference dataset
# -------------------------------------
class InferenceDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None):
        self.images = sorted([
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        self.masks = None
        if masks_dir and os.path.isdir(masks_dir):
            self.masks = sorted([
                os.path.join(masks_dir, f)
                for f in os.listdir(masks_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            assert len(self.images) == len(self.masks), "Image/mask count mismatch"
        self.image_tf = torchvision.transforms.ToTensor()
        self.mask_tf  = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.image_tf(img)

        mask = None
        if self.masks:
            m = Image.open(self.masks[idx])
            mask = self.mask_tf(m)

        fname = os.path.basename(img_path)
        return img, mask, fname

# -------------------------------------
# 4. Dice coefficient (for evaluation)
# -------------------------------------
def dice_coef(pred, target, eps=1e-6):
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return ((2*inter + eps)/(union + eps)).mean().item()

# -------------------------------------
# 5. Run Inference
# -------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # DataLoader
    ds     = InferenceDataset(IMAGES_DIR, MASKS_DIR)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = EfficientNetUNet(n_classes=1)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.to(device).eval()

    dices = []
    with torch.no_grad():
        for imgs, masks, fnames in loader:
            imgs   = imgs.to(device)
            logits = model(imgs)
            probs  = torch.sigmoid(logits)
            preds  = (probs > THRESHOLD).float()

            # Save predictions
            for b in range(preds.size(0)):
                arr = (preds[b,0].cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(arr).save(os.path.join(OUTPUT_DIR, fnames[b]))

            # Compute Dice if mask present
            if masks is not None:
                dices.append(dice_coef(preds, masks.to(device)))

    if dices:
        print(f"Average Dice over {len(dices)} images: {np.mean(dices):.4f}")
    print("Inference complete. Predictions saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
