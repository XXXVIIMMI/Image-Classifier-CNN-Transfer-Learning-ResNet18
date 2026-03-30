from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src import config


def get_transforms(is_train: bool):
    """Different augmentations for train vs val."""

    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   # ImageNet mean
                std=[0.229, 0.224, 0.225]      # ImageNet std
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def create_dataloaders(data_dir: Path):

    train_dataset = datasets.ImageFolder(
        root=data_dir / "train",
        transform=get_transforms(is_train=True)
    )

    val_dataset = datasets.ImageFolder(
        root=data_dir / "val",
        transform=get_transforms(is_train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    class_names = train_dataset.classes   # e.g. ['bird', 'cat', 'dog']
    print(f"[data_setup] Classes found : {class_names}")
    print(f"[data_setup] Train size    : {len(train_dataset)}")
    print(f"[data_setup] Val size      : {len(val_dataset)}")

    return train_loader, val_loader, class_names
