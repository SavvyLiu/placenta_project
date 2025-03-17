import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from PlacentaDataset import PlacentaDataset

# Data augmentation with Albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2


def train_smp():
    # -----------------------------
    # 1. Hyperparameters & Setup
    # -----------------------------
    images_dir = "data/images"
    masks_dir = "data/masks"
    batch_size = 4  # Increased batch size if VRAM permits
    num_epochs = 100
    lr = 1e-4
    train_val_split = 0.8  # 80% training, 20% validation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # 2. Data Augmentation & Transforms
    # -----------------------------
    # Training augmentation and normalization
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Use Affine instead of ShiftScaleRotate to avoid warning
        A.Affine(translate_percent=0.0625, scale=0.1, rotate=15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    # Validation: only normalization and conversion to tensor
    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # -----------------------------
    # 3. Create Dataset, Split & DataLoader
    # -----------------------------
    full_dataset = PlacentaDataset(images_dir, masks_dir, transform=train_transform)
    total_size = len(full_dataset)
    train_size = int(train_val_split * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Change transform for validation dataset if applicable
    if hasattr(val_dataset, 'dataset'):
        val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

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
    # Removed verbose parameter as it is deprecated.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                     patience=5)
    # Updated usage: using torch.amp instead of torch.cuda.amp.
    scaler = torch.amp.GradScaler()

    # -----------------------------
    # 6. Training Loop with Validation, Early Stopping & Checkpointing
    # -----------------------------
    best_val_loss = float('inf')
    patience = 10  # epochs to wait for improvement
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            # Use torch.amp.autocast with device type 'cuda'
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = combined_loss(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
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
            print("Validation loss improved. Model saved as best_smp_unet_placenta.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    train_smp()
