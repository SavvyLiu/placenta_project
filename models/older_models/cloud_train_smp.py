import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from PlacentaDataset import PlacentaDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os


def train_smp(use_subset=False):
    # -----------------------------
    # 1. Hyperparameters & Setup
    # -----------------------------
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root
    project_dir = os.path.dirname(script_dir)
    # Construct paths relative to the project root
    images_dir = os.path.join(project_dir, "data", "images")
    masks_dir = os.path.join(project_dir, "data", "masks")
    batch_size = 1
    num_epochs = 50
    lr = 1e-4
    train_val_split = 0.8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable cudnn benchmark for potential speedup
    torch.backends.cudnn.benchmark = True

    # -----------------------------
    # 2. Data Augmentation & Transforms
    # -----------------------------
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=0.0625, scale=0.1, rotate=15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # -----------------------------
    # 3. Create Dataset, Split & DataLoader
    # -----------------------------
    full_dataset = PlacentaDataset(images_dir, masks_dir, transform=train_transform, use_subset=use_subset)
    total_size = len(full_dataset)
    train_size = int(train_val_split * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    if hasattr(val_dataset, 'dataset'):
        val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True)

    # -----------------------------
    # 4. Define the Model
    # -----------------------------
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    model.to(device)

    # -----------------------------
    # 5. Define Loss, Optimizer & Scheduler
    # -----------------------------
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()
    lambda_dice = 1.0
    lambda_bce = 1.0

    def combined_loss(pred, target):
        return lambda_bce * bce_loss(pred, target) + lambda_dice * dice_loss(pred, target)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                     patience=5)
    scaler = torch.amp.GradScaler()

    # -----------------------------
    # 6. Training Loop with Validation, Early Stopping & Checkpointing
    # -----------------------------
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = combined_loss(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = combined_loss(outputs, masks)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_smp_unet_placenta.pth")
            print("Validation loss improved. Model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    use_subset = input("Use subset of 4 images for training? (y/n): ").lower() == 'y'
    train_smp(use_subset)
