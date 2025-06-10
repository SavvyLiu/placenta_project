import argparse

from models.efficicentnet_train_smp import train_efficientnet
from models.regnet_train_smp import train_regnet
from models.train_UNET_smp import train_smp
from models.ViT_train_smp import train_vit

def main():
    parser = argparse.ArgumentParser(
        description="Train one or more placenta segmentation models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["efficientnet", "regnet", "unet", "vit"],
        default=["efficientnet", "regnet", "unet", "vit"],
        help="Which model(s) to train"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs (for scripts that accept it)"
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=0,
        help="Use only the first N images/masks for quick debugging (0 = full dataset)"
    )
    parser.add_argument(
        "--lr-patience", 
        type=int,
        default=5,
        help="Learning rate scheduler patience (epochs to wait before reducing LR)"
    )
    parser.add_argument(
        "--lr-factor",
        type=float, 
        default=0.5,
        help="Learning rate reduction factor (multiply LR by this when reducing)"
    )
    args = parser.parse_args()

    for m in args.models:
        print(f"\n=== Training {m} ===")
        if m == "efficientnet":
            # train_efficientnet expects (numofepochs: str, subset_size: int, lr_patience: int, lr_factor: float)
            train_efficientnet(str(args.epochs), subset_size=args.subset_size, lr_patience=args.lr_patience, lr_factor=args.lr_factor)
        elif m == "regnet":
            # train_regnet(numofepochs: str, subset_size: int, lr_patience: int, lr_factor: float)
            train_regnet(str(args.epochs), subset_size=args.subset_size, lr_patience=args.lr_patience, lr_factor=args.lr_factor)
        elif m == "unet":
            # train_smp(num_epochs: int, subset_size: int, lr_patience: int, lr_factor: float)
            train_smp(args.epochs, subset_size=args.subset_size, lr_patience=args.lr_patience, lr_factor=args.lr_factor)
        elif m == "vit":
            # train_vit expects (num_epochs: int, subset_size: int, lr_patience: int, lr_factor: float)
            train_vit(args.epochs, subset_size=args.subset_size, lr_patience=args.lr_patience, lr_factor=args.lr_factor)
        print(f"--- {m} done ---")

if __name__ == "__main__":
    main()
