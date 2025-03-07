import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PlacentaDataset import PlacentaDataset
def train_unet():
    # -----------------------------
    # 1. Hyperparameters & Setup
    # -----------------------------
    images_dir = "data/images"
    masks_dir = "data/masks"
    batch_size = 2
    lr = 1e-4
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -----------------------------
    # 2. Create Dataset & DataLoader
    # -----------------------------
    train_dataset = PlacentaDataset(images_dir, masks_dir)
    print("test")

train_unet()


"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # -----------------------------
    # 3. Initialize Model, Loss, Optimizer
    # -----------------------------
    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Good for binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # -----------------------------
    # 4. Training Loop
    # -----------------------------
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)  # (B, 3, H, W)
            masks = masks.to(device)    # (B, 1, H, W)
            
            # Forward pass
            outputs = model(images)     # (B, 1, H, W)
            loss = criterion(outputs, masks)
            
            # Backward & optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * images.size(0)
        
        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    # -----------------------------
    # 5. Save the Trained Model
    # -----------------------------
    torch.save(model.state_dict(), "unet_placenta.pth")
    print("Model saved as unet_placenta.pth")

if __name__ == "__main__":
    train_unet()
"""