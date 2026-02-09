import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import torchvision
from torchvision.models import regnet_y_400mf, RegNet_Y_400MF_Weights
import segmentation_models_pytorch as smp
from models.PlacentaDataset import PlacentaDataset
from utilities.metrics import SegmentationMetrics
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Define a custom U-Net using RegNet_Y_400MF as the encoder.
class RegNetUNet(nn.Module):
    def __init__(self, n_classes=3):
        super(RegNetUNet, self).__init__()
        # Load the pretrained RegNet_Y_400MF model from torchvision
        weights = RegNet_Y_400MF_Weights.IMAGENET1K_V1
        self.encoder = regnet_y_400mf(weights=weights)
        # Combine stem and trunk_output for feature extraction
        self.encoder_features = nn.Sequential(
            self.encoder.stem,
            self.encoder.trunk_output
        )  # output shape: (B, 440, H/32, W/32)

        # Build a decoder with correct number of upsampling steps
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(440, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        # Final 1x1 convolution to get the desired number of output classes
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Extract features from the encoder
        features = self.encoder_features(x)  # shape: (B, 440, H/32, W/32)
        x = self.decoder(features)  # progressively upsample the feature maps
        x = self.final_conv(x)
        return x


def train_regnet(numofepochs, subset_size=0, lr_patience=5, lr_factor=0.5, batch_size=16, augment=True, early_stopping_patience=10):
    # -------------------------------------
    # 1. Hyperparameters & Setup
    # -------------------------------------
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root
    project_dir = os.path.dirname(script_dir)
    # Construct paths relative to the project root
    images_dir = os.path.join(project_dir, "data", "images")
    masks_dir = os.path.join(project_dir, "data", "masks")
    num_epochs = int(numofepochs)
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    logger.info(f"Batch size: {batch_size}, Augmentation: {augment}, Early Stopping Patience: {early_stopping_patience}")

    # -------------------------------------
    # 2. Create Dataset & DataLoader
    # -------------------------------------
    # Use target_size=512 to reduce memory usage (especially for large TIF files)
    dataset = PlacentaDataset(images_dir, masks_dir, subset_size=subset_size, augment=augment, target_size=512)
    
    # Split into train and validation (80-20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    logger.info(f"Dataset: {len(dataset)} total, {train_size} train, {val_size} val")

    # -------------------------------------
    # 3. Instantiate the Model with the new backbone
    # -------------------------------------
    model = RegNetUNet(n_classes=3)
    model.to(device)

    # -------------------------------------
    # 4. Define Loss and Optimizer
    # -------------------------------------
    # Use CrossEntropyLoss for multi-class segmentation
    # Combined with Dice Loss for better segmentation quality
    ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # ignore background class
    dice_loss = smp.losses.DiceLoss(mode='multiclass', classes=3)

    def combined_loss(pred, target):
        # pred shape: (B, 3, H, W)
        # target shape: (B, H, W) with values 0, 1, 2
        ce = ce_loss(pred, target)
        dice = dice_loss(pred, target)
        return ce + dice

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, factor=lr_factor)
    scaler = GradScaler()  # For mixed precision training
    
    # Metrics tracker
    metrics_tracker = SegmentationMetrics(num_classes=3, class_names={0: 'background', 1: 'fetal', 2: 'maternal'})
    
    # Validation function
    def validate(model, val_loader, device):
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                with autocast():  # Mixed precision inference
                    outputs = model(images)
                    loss = combined_loss(outputs, masks)
                
                val_loss += loss.item() * images.size(0)
                all_preds.append(torch.argmax(outputs, dim=1).cpu())
                all_targets.append(masks.cpu())
        
        val_loss /= val_size
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = metrics_tracker.compute_all(all_preds, all_targets)
        
        return val_loss, metrics

    # -------------------------------------
    # 5. Training Loop with Early Stopping
    # -------------------------------------
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_dataloader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            
            # Mixed precision training
            with autocast():
                outputs = model(images)
                loss = combined_loss(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * images.size(0)

        epoch_loss /= train_size
        
        # Validation
        val_loss, val_metrics = validate(model, val_dataloader, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        logger.info(f"  Val IoU (mean): {val_metrics['iou']['mean_iou']:.4f}, Val Dice (mean): {val_metrics['dice']['mean_dice']:.4f}")
        logger.info(f"  Per-class - IoU: {val_metrics['iou']}")
        logger.info(f"  Per-class - Dice: {val_metrics['dice']}")
        
        # Track best validation loss and save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            logger.info(f"  ✓ New best validation loss: {best_val_loss:.4f}")
            
            # Save best model
            save_dir = os.path.join(project_dir, "trained_models")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "regnet_unet_placenta_best.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✓ Best model saved to {save_path}")
        else:
            patience_counter += 1
            logger.info(f"  ✗ No improvement. Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs!")
                break

    # Save final model
    save_dir = os.path.join(project_dir, "trained_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "regnet_unet_placenta_final.pth")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Final model saved as {save_path}")
    logger.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train RegNet U-Net for 3-class placenta segmentation")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (A40: default 16)")
    parser.add_argument("--subset-size", type=int, default=0, help="Subset size (0 = full dataset)")
    parser.add_argument("--lr-patience", type=int, default=5, help="Learning rate scheduler patience")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="Learning rate reduction factor")
    parser.add_argument("--augment", action="store_true", default=True, help="Apply on-the-fly augmentation")
    parser.add_argument("--no-augment", dest="augment", action="store_false", help="Disable augmentation")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience")
    
    args = parser.parse_args()
    train_regnet(
        args.epochs, 
        subset_size=args.subset_size, 
        lr_patience=args.lr_patience, 
        lr_factor=args.lr_factor,
        batch_size=args.batch_size,
        augment=args.augment,
        early_stopping_patience=args.early_stopping_patience
    )
