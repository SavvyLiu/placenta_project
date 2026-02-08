import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_h_14, ViT_H_14_Weights
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from models.PlacentaDataset import PlacentaDataset
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ViT_UNet_Flexible(nn.Module):
    def __init__(self, n_classes=3, pretrained=True):
        super().__init__()
        # 1) load ViT-H-14 and strip its head
        weights = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None
        self.vit = vit_h_14(weights=weights)
        self.vit.heads = nn.Identity()

        # 2) pull out the pieces we need
        self.conv_proj      = self.vit.conv_proj
        self.class_token    = self.vit.class_token
        encoder              = self.vit.encoder
        self.orig_pos_embed = encoder.pos_embedding       # (1,1+37*37,D)
        self.encoder_blocks = encoder.layers              # nn.Sequential of EncoderBlock
        self.encoder_norm   = encoder.ln                  # final LayerNorm

        # 3) record sizes
        self.patch_size = self.vit.patch_size             # 14
        self.embed_dim  = self.vit.hidden_dim             # ≈1024

        # 4) simple ConvTranspose2d decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,  64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d( 64,  32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

        # 5) register ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def forward(self, x):
        B, C, H, W = x.shape
        ps = self.patch_size

        # — pad to a multiple of patch_size
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        H_pad, W_pad = x.shape[-2:]

        # — normalize
        x = (x - self.mean) / self.std

        # — patch-embed
        x = self.conv_proj(x)                 # (B, D, H_pad/14, W_pad/14)
        B, D, h, w = x.shape
        tokens = x.flatten(2).transpose(1,2)  # (B, h*w, D)

        # — prepend CLS token
        cls    = self.class_token.expand(B, -1, -1)  # (B,1,D)
        tokens = torch.cat([cls, tokens], dim=1)     # (B,1+h*w,D)

        # — interpolate pos_embedding from 37×37 → h×w
        old_pe    = self.orig_pos_embed               # (1,1+37*37,D)
        cls_pe    = old_pe[:, :1, :].clone()          # (1,1,D)
        grid_pe   = old_pe[:, 1:, :].transpose(1,2)   # (1,D,37*37)
        grid_pe   = grid_pe.view(1, D, 37, 37)        # (1,D,37,37)
        resized   = F.interpolate(
                        grid_pe,
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False
                    )                              # (1,D,h,w)
        resized   = resized.flatten(2).transpose(1,2) # (1,h*w,D)
        pos_embed = torch.cat([cls_pe, resized], dim=1)  # (1,1+h*w,D)

        # — encode with manual blocks + norm
        x_enc = tokens + pos_embed
        for blk in self.encoder_blocks:
            x_enc = blk(x_enc)
        x_enc = self.encoder_norm(x_enc)  # (B,1+h*w,D)

        # — reshape back to spatial
        x_enc = x_enc[:, 1:, :].transpose(1,2).view(B, D, h, w)

        # — decode & crop to original H×W
        x_dec = self.decoder(x_enc)
        x_out = self.final_conv(x_dec)
        return x_out[..., :H, :W]

def train_vit(num_epochs: int, subset_size=0, lr_patience=5, lr_factor=0.5):
    # Determine project directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    images_dir = os.path.join(project_dir, "data", "images")
    masks_dir  = os.path.join(project_dir, "data", "masks")
    batch_size = 1
    lr         = 1e-4
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")

    ds = PlacentaDataset(images_dir, masks_dir, subset_size=subset_size)
    
    # Split into train and validation (80-20 split)
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_dataset, val_dataset = random_split(ds, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Dataset: {len(ds)} total, {train_size} train, {val_size} val")

    model     = ViT_UNet_Flexible(n_classes=3).to(device)
    ce_loss   = nn.CrossEntropyLoss(ignore_index=0)  # ignore background class
    dice_loss = smp.losses.DiceLoss(mode='multiclass', classes=3)
    def loss_fn(p, t): return ce_loss(p, t) + dice_loss(p, t)
    opt       = torch.optim.Adam(model.parameters(), lr=lr)

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=lr_patience, factor=lr_factor)
    
    # Validation function
    def validate(model, val_loader, device):
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                loss = loss_fn(preds, masks)
                val_loss += loss.item() * imgs.size(0)
        
        val_loss /= val_size
        return val_loss

    best_val_loss = float('inf')
    prev = float("inf")
    for epoch in range(num_epochs):
        model.train()
        total = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            preds = model(imgs)
            loss  = loss_fn(preds, masks)
            loss.backward()
            opt.step()
            total += loss.item() * imgs.size(0)

        avg = total / train_size
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Step the scheduler
        scheduler.step(val_loss)
        current_lr = opt.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{num_epochs}  Train Loss={avg:.4f}  Val Loss={val_loss:.4f}  LR={current_lr:.6f}")
        
        # Track best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"  -> New best validation loss: {best_val_loss:.4f}")
        
        prev = avg

    # ensure trained_models directory exists
    save_dir = os.path.join(project_dir, "trained_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "vit_unet_placenta_flexible.pth")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved {save_path}")

if __name__ == "__main__":
    n = int(input("Enter number of epochs: "))
    subset_size = int(input("Enter subset size (0 = full dataset): ") or "0")
    lr_patience = int(input("Enter LR scheduler patience (default 5): ") or "5")
    lr_factor = float(input("Enter LR reduction factor (default 0.5): ") or "0.5")
    train_vit(n, subset_size=subset_size, lr_patience=lr_patience, lr_factor=lr_factor)
