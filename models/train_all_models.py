import argparse

from models.efficicentnet_train_smp import train_efficientnet
from models.regnet_train_smp import train_regnet
from models.train_UNET_smp import train_smp
from models.ViT_train_smp import train_vit

import argparse

from models.efficicentnet_train_smp import train_efficientnet
from models.regnet_train_smp import train_regnet
from models.train_UNET_smp import train_smp
from models.ViT_train_smp import train_vit

def main():
    parser = argparse.ArgumentParser(
        description="Train one or more placenta segmentation models optimized for A40 GPU"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["efficientnet", "regnet", "unet", "vit"],
        default=["efficientnet"],
        help="Which model(s) to train"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (A40: recommended 16-32)"
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
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Apply on-the-fly augmentation"
    )
    parser.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="Disable on-the-fly augmentation"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs to wait before stopping if no improvement)"
    )
    args = parser.parse_args()

    for m in args.models:
        print(f"\n{'='*60}")
        print(f"Training {m.upper()} with:")
        print(f"  Epochs: {args.epochs}, Batch Size: {args.batch_size}")
        print(f"  Augmentation: {args.augment}, Early Stopping: {args.early_stopping_patience}")
        print(f"{'='*60}")
        
        if m == "efficientnet":
            train_efficientnet(
                args.epochs, 
                subset_size=args.subset_size, 
                lr_patience=args.lr_patience, 
                lr_factor=args.lr_factor,
                batch_size=args.batch_size,
                augment=args.augment,
                early_stopping_patience=args.early_stopping_patience
            )
        elif m == "regnet":
            train_regnet(
                args.epochs, 
                subset_size=args.subset_size, 
                lr_patience=args.lr_patience, 
                lr_factor=args.lr_factor,
                batch_size=args.batch_size,
                augment=args.augment,
                early_stopping_patience=args.early_stopping_patience
            )
        elif m == "unet":
            train_smp(
                args.epochs, 
                subset_size=args.subset_size, 
                lr_patience=args.lr_patience, 
                lr_factor=args.lr_factor,
                batch_size=args.batch_size,
                augment=args.augment,
                early_stopping_patience=args.early_stopping_patience
            )
        elif m == "vit":
            train_vit(
                args.epochs, 
                subset_size=args.subset_size, 
                lr_patience=args.lr_patience, 
                lr_factor=args.lr_factor,
                batch_size=args.batch_size,
                augment=args.augment,
                early_stopping_patience=args.early_stopping_patience
            )
        print(f"✓ {m.upper()} training completed!")


if __name__ == "__main__":
    main()
