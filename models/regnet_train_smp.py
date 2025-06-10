import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import regnet_y_400mf, RegNet_Y_400MF_Weights
import segmentation_models_pytorch as smp  # still using its loss if desired
from models.PlacentaDataset import PlacentaDataset
import os


# Define a custom U-Net using RegNet_Y_400MF as the encoder.
class RegNetUNet(nn.Module):
    def __init__(self, n_classes=1):
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


def train_regnet(numofepochs, use_subset=False):
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
    dataset = PlacentaDataset(images_dir, masks_dir, use_subset=use_subset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # -------------------------------------
    # 3. Instantiate the Model with the new backbone
    # -------------------------------------
    model = RegNetUNet(n_classes=1)
    model.to(device)

    # -------------------------------------
    # 4. Define Loss and Optimizer
    # -------------------------------------
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()

    def combined_loss(pred, target):
        return bce_loss(pred, target) + dice_loss(pred, target)

    optimizer = optim.Adam(model.parameters(), lr=lr)

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
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Improvement: {prev_loss - epoch_loss:.4f}")

    # -------------------------------------
    # 6. Save the Trained Model
    # -------------------------------------
    # ensure trained_models directory exists
    save_dir = os.path.join(project_dir, "trained_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "regnet_unet_placenta.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")


if __name__ == "__main__":
    numofepochs = input("Please enter number of Epochs: ")
    use_subset = input("Use subset of 4 images for training? (y/n): ").lower() == 'y'
    train_regnet(numofepochs, use_subset)
