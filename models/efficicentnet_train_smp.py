import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import segmentation_models_pytorch as smp  # still using its loss if desired
from models.PlacentaDataset import PlacentaDataset


# Define a custom U-Net using EfficientNet_V2_L as the encoder.
class EfficientNetUNet(nn.Module):
    def __init__(self, n_classes=1):
        super(EfficientNetUNet, self).__init__()
        # Load the pretrained EfficientNet_V2_L model from torchvision
        weights = torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
        self.encoder = torchvision.models.efficientnet_v2_l(weights=weights)
        # Remove the classifier and use the feature extractor
        self.encoder_features = self.encoder.features  # output shape: (B, 1280, H/32, W/32)

        # Build a decoder with correct number of upsampling steps
        self.decoder = nn.Sequential(
            # First upsampling: 1280 -> 512
            nn.ConvTranspose2d(1280, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            # Second upsampling: 512 -> 256
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            # Third upsampling: 256 -> 128
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            # Fourth upsampling: 128 -> 64
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            # Fifth upsampling: 64 -> 32
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        # Final 1x1 convolution to get the desired number of output classes
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Extract features from the encoder
        features = self.encoder_features(x)  # shape: (B, 1280, H/32, W/32)
        x = self.decoder(features)  # progressively upsample the feature maps
        x = self.final_conv(x)
        return x


def train_efficientnet(numofepochs, subset_size=0, lr_patience=5, lr_factor=0.5):
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
    batch_size = 1
    num_epochs = int(numofepochs)
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------
    # 2. Create Dataset & DataLoader
    # -------------------------------------
    dataset = PlacentaDataset(images_dir, masks_dir, subset_size=subset_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # -------------------------------------
    # 3. Instantiate the Model with the new backbone
    # -------------------------------------
    model = EfficientNetUNet(n_classes=1)
    model.to(device)

    # -------------------------------------
    # 4. Define Loss and Optimizer
    # -------------------------------------
    # Using a combination of Dice Loss and Binary Cross-Entropy Loss.
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()

    def combined_loss(pred, target):
        return bce_loss(pred, target) + dice_loss(pred, target)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, factor=lr_factor)

    # -------------------------------------
    # 5. Training Loop
    # -------------------------------------
    epoch_loss = 5.0
    for epoch in range(num_epochs):
        prev_loss = epoch_loss
        model.train()
        epoch_loss = 0.0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # shape: (B, 1, H, W)
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)

        epoch_loss /= len(dataset)
        
        # Step the scheduler
        scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Improvement: {prev_loss - epoch_loss:.4f}")
        if epoch > 0:
            print(f"  LR: {current_lr:.6f}")

    # -------------------------------------
    # 6. Save the Trained Model
    # -------------------------------------
    # ensure trained_models directory exists
    save_dir = os.path.join(project_dir, "trained_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "efficientnet_unet_placenta.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")


if __name__ == "__main__":
    numofepochs = input("Please enter number of Epochs: ")
    subset_size = int(input("Enter subset size (0 = full dataset): "))
    lr_patience = int(input("Enter LR scheduler patience (default 5): ") or "5")
    lr_factor = float(input("Enter LR reduction factor (default 0.5): ") or "0.5")
    train_efficientnet(numofepochs, subset_size=subset_size, lr_patience=lr_patience, lr_factor=lr_factor)
