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
        "--subset",
        action="store_true",
        help="Use subset of 4 images for quick debugging"
    )
    args = parser.parse_args()

    for m in args.models:
        print(f"\n=== Training {m} ===")
        if m == "efficientnet":
            # train_efficientnet expects (numofepochs: str, use_subset: bool)
            train_efficientnet(str(args.epochs), use_subset=args.subset)
        elif m == "regnet":
            # train_regnet(numofepochs: str, use_subset: bool)
            train_regnet(str(args.epochs), use_subset=args.subset)
        elif m == "unet":
            # train_smp(use_subset: bool) – epochs is fixed inside
            train_smp(use_subset=args.subset)
        elif m == "vit":
            # train_vit expects (num_epochs: int, use_subset: bool)
            train_vit(args.epochs, use_subset=args.subset)
        print(f"--- {m} done ---")

if __name__ == "__main__":
    main()
