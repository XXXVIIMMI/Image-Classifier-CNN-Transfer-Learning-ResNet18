import torch
import matplotlib.pyplot as plt
from pathlib import Path


def save_model(model, save_dir: Path, filename: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    torch.save(model.state_dict(), save_path)
    print(f"[utils] Model saved to {save_path}")


def load_model(model, load_path: Path, device):
    model.load_state_dict(torch.load(load_path, map_location=device))
    print(f"[utils] Model loaded from {load_path}")
    return model


def plot_results(results: dict, save_path: str = "results.png"):
 
    epochs = range(1, len(results["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, results["train_loss"], label="train loss")
    ax1.plot(epochs, results["val_loss"],   label="val loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    # Accuracy
    ax2.plot(epochs, results["train_acc"], label="train acc")
    ax2.plot(epochs, results["val_acc"],   label="val acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[utils] Plot saved to {save_path}")
    plt.show()
