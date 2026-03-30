from pathlib import Path
from torchvision import datasets, transforms

from src import config


def download_and_prepare():
    """
    Downloads CIFAR-10 via torchvision (no login needed).
    Keeps only 3 classes: bird, cat, dog.
    Saves images as .png into data/train/ and data/val/.
    """

    print("[download] Downloading CIFAR-10...")

    TARGET_CLASSES = {
        2: "bird",
        3: "cat",
        5: "dog",
    }

    raw_transform = transforms.ToTensor()

    train_raw = datasets.CIFAR10(
        root="data/raw", train=True,  download=True, transform=raw_transform
    )
    val_raw = datasets.CIFAR10(
        root="data/raw", train=False, download=True, transform=raw_transform
    )

    _save_subset(train_raw, TARGET_CLASSES, config.DATA_DIR / "train")
    _save_subset(val_raw,   TARGET_CLASSES, config.DATA_DIR / "val")

    print("[download] Done. Data is ready.")
    _print_summary(config.DATA_DIR)


def _save_subset(dataset, target_classes: dict, save_dir: Path):

    # Create class subfolders
    for class_name in target_classes.values():
        (save_dir / class_name).mkdir(parents=True, exist_ok=True)

    counters = {name: 0 for name in target_classes.values()}

    for idx in range(len(dataset)):
        img_tensor, label = dataset[idx]

        if label not in target_classes:
            continue

        class_name = target_classes[label]

        # Convert tensor -> PIL Image -> save as PNG
        img_pil = transforms.ToPILImage()(img_tensor)
        save_path = save_dir / class_name / f"{class_name}_{counters[class_name]:04d}.png"
        img_pil.save(save_path)

        counters[class_name] += 1

    print(f"[download] Saved to {save_dir}: {counters}")


def _print_summary(data_dir: Path):
    print("\n[download] Final folder structure:")
    for split in ["train", "val"]:
        for class_folder in sorted((data_dir / split).iterdir()):
            count = len(list(class_folder.glob("*.png")))
            print(f"  {split}/{class_folder.name}/  ->  {count} images")
