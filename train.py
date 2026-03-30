import torch
from src import config
from src.download_data import download_and_prepare
from src.data_setup    import create_dataloaders
from src.model         import create_model
from src.engine        import train
from src.utils         import save_model, plot_results


def main():

    # --- Step 0: Download data (only runs if data/train/ doesn't exist yet) ---
    if not (config.DATA_DIR / "train").exists() or \
       not any((config.DATA_DIR / "train").iterdir()):
        print("[train] Data not found. Downloading...")
        download_and_prepare()
    else:
        print("[train] Data already exists. Skipping download.")

    print("\n=== Starting Training ===")
    print(f"  Device      : {config.DEVICE}")
    print(f"  Epochs      : {config.NUM_EPOCHS}")
    print(f"  Batch size  : {config.BATCH_SIZE}")
    print(f"  LR          : {config.LEARNING_RATE}")
    print(f"  Num classes : {config.NUM_CLASSES}")

    # --- Step 1: Data ---
    train_loader, val_loader, class_names = create_dataloaders(config.DATA_DIR)

    # --- Step 2: Model ---
    model = create_model(num_classes=config.NUM_CLASSES, freeze_backbone=False)
    model = model.to(config.DEVICE)

    # --- Step 3: Loss and Optimizer ---
    loss_fn = torch.nn.CrossEntropyLoss()

    # Differential LRs: head trains fast, backbone fine-tunes slowly
    optimizer = torch.optim.Adam([
        {"params": model.fc.parameters(),
         "lr": config.LEARNING_RATE},
        {"params": [p for name, p in model.named_parameters()
                    if not name.startswith("fc")],
         "lr": config.LEARNING_RATE * 0.1}
    ])

    # --- Step 4: Train ---
    results = train(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        loss_fn      = loss_fn,
        optimizer    = optimizer,
        epochs       = config.NUM_EPOCHS,
        device       = config.DEVICE
    )

    # --- Step 5: Save model and plot results ---
    save_model(model, config.MODEL_DIR, "resnet18_classifier.pth")
    plot_results(results, save_path="results.png")

    print("\n=== Training Complete ===")
    print(f"  Best val acc : {max(results['val_acc']):.4f}")
    print(f"  Model saved  : models/resnet18_classifier.pth")
    print(f"  Plot saved   : results.png")


if __name__ == "__main__":
    main()
