import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import segmentation_models_pytorch as smp
from PlacentaDataset import PlacentaDataset
from sklearn.model_selection import KFold


def combined_loss_fn(pred, target, dice_loss, bce_loss):
    """Combine Dice loss and BCE loss."""
    return bce_loss(pred, target) + dice_loss(pred, target)


def train_fold(fold, train_idx, val_idx, dataset, num_epochs, batch_size, lr, device):
    # Create subsets for the current fold
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Initialize model for this fold
    model = smp.Unet(
        encoder_name="resnet34",  # encoder, e.g., resnet34
        encoder_weights="imagenet",  # pre-trained weights for encoder
        in_channels=3,  # input channels (RGB)
        classes=1  # binary segmentation
    )
    model.to(device)

    # Define loss functions and optimizer
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    print(f"Training fold {fold}...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss_fn(outputs, masks, dice_loss, bce_loss)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_subset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = combined_loss_fn(outputs, masks, dice_loss, bce_loss)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_subset)
        print(f"Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Optionally, save the best model for this fold
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"smp_unet_placenta_fold{fold}.pth")
            print(f"Fold {fold}: New best model saved with val loss {best_val_loss:.4f}")

    return best_val_loss


def cross_validate():
    # -----------------------------
    # Hyperparameters & Setup
    # -----------------------------
    images_dir = "../data/images"
    masks_dir = "../data/masks"
    num_epochs = 50
    batch_size = 2
    lr = 1e-4
    num_folds = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the dataset
    dataset = PlacentaDataset(images_dir, masks_dir)

    # Initialize KFold cross-validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        val_loss = train_fold(fold, train_idx, val_idx, dataset, num_epochs, batch_size, lr, device)
        fold_results.append(val_loss)

    print("Cross-validation results (best val loss per fold):", fold_results)
    print("Average validation loss:", sum(fold_results) / len(fold_results))


if __name__ == "__main__":
    cross_validate()
