#!/usr/bin/env python3
import os
import sys

# Ensure the project root is on the import path (so `models` can be found)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import torch
import numpy as np
import argparse
from models.efficicentnet_train_smp import EfficientNetUNet


def load_model(weights_path, device):
    model = EfficientNetUNet(n_classes=1)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model


def infer_and_save(model, device, input_img_path, mask_out_path, annot_out_path, threshold=0.5):
    # 1) load and preprocess
    img_bgr = cv2.imread(input_img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load input image: {input_img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = (
        torch.from_numpy(img_rgb)
             .permute(2, 0, 1)
             .unsqueeze(0)
             .to(device)
    )

    # 2) inference
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    # 3) threshold -> uint8 mask
    mask = (prob > threshold).astype(np.uint8) * 255
    os.makedirs(os.path.dirname(mask_out_path), exist_ok=True)
    cv2.imwrite(mask_out_path, mask)
    print(f"Mask saved to {mask_out_path}")

    # 4) find contours & annotate
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = img_bgr.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue
        cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), 2)
    os.makedirs(os.path.dirname(annot_out_path), exist_ok=True)
    cv2.imwrite(annot_out_path, annotated)
    print(f"Annotated image saved to {annot_out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Quick test inference")
    p.add_argument("--model",     required=True, help="Path to .pth weights")
    p.add_argument("--input",     required=True, help="Path to input image")
    p.add_argument("--mask_out",  required=True, help="Where to save predicted mask")
    p.add_argument("--annot_out", required=True, help="Where to save contour overlay")
    p.add_argument("--threshold", type=float, default=0.5, help="Binary cutoff")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    infer_and_save(model, device,
                   args.input,
                   args.mask_out,
                   args.annot_out,
                   threshold=args.threshold)
