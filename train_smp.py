import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from PlacentaDataset import PlacentaDataset
# Make sure PlacentaDataset is defined or imported here
# from dataset import PlacentaDataset

def train_smp():
    # -----------------------------
    # 1. Hyperparameters & Setup
    # -----------------------------
    images_dir = "data/images"
    masks_dir = "data/masks"
    batch_size = 2
    num_epochs = 35
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -----------------------------
    # 2. Create Dataset & DataLoader
    # -----------------------------
    dataset = PlacentaDataset(images_dir, masks_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # -----------------------------
    # 3. Define the Model
    # -----------------------------
    # Here we use a U-Net with a ResNet34 backbone pretrained on ImageNet.
    model = smp.Unet(
        encoder_name="resnet34",      # choose encoder, e.g., resnet34
        encoder_weights="imagenet",     # use pre-trained weights for encoder
        in_channels=3,                  # model input channels (RGB)
        classes=1                     # number of output channels (binary segmentation)
    )
    model.to(device)
    
    # -----------------------------
    # 4. Define Loss and Optimizer
    # -----------------------------
    # Use a combination of Dice Loss and Binary Cross-Entropy Loss
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()
    
    def combined_loss(pred, target):
        return bce_loss(pred, target) + dice_loss(pred, target)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # -----------------------------
    # 5. Training Loop
    # -----------------------------
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
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Difference: {dice_loss(outputs, masks):.4f}, Difference Loss: {prev_loss - epoch_loss}")
    
    # -----------------------------
    # 6. Save the Trained Model
    # -----------------------------
    torch.save(model.state_dict(), "smp_unet_placenta.pth")
    print("Model saved as smp_unet_placenta.pth")

if __name__ == "__main__":
    train_smp()
