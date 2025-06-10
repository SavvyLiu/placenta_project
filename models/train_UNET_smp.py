import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from models.PlacentaDataset import PlacentaDataset
import os
# Make sure PlacentaDataset is defined or imported here
# from dataset import PlacentaDataset

def train_smp(num_epochs=100, use_subset=False, lr_patience=5, lr_factor=0.5):
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
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -----------------------------
    # 2. Create Dataset & DataLoader
    # -----------------------------
    dataset = PlacentaDataset(images_dir, masks_dir, use_subset=use_subset)
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
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, factor=lr_factor)
    
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
        
        # Step the scheduler
        scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Difference: {dice_loss(outputs, masks):.4f}, Difference Loss: {prev_loss - epoch_loss}")
        if epoch > 0:
            print(f"  LR: {current_lr:.6f}")
    
    # -----------------------------
    # 6. Save the Trained Model
    # -----------------------------
    # ensure trained_models directory exists
    save_dir = os.path.join(project_dir, "trained_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "smp_unet_placenta.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")

if __name__ == "__main__":
    num_epochs = int(input("Please enter number of Epochs: "))
    subset_size = int(input("Enter subset size (0 = full dataset): ") or "0")
    lr_patience = int(input("Enter LR scheduler patience (default 5): ") or "5")
    lr_factor = float(input("Enter LR reduction factor (default 0.5): ") or "0.5")
    train_smp(num_epochs, use_subset=subset_size, lr_patience=lr_patience, lr_factor=lr_factor)
