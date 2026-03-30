import argparse
import torch
from PIL import Image
from torchvision import transforms

from src import config
from src.model import create_model
from src.utils import load_model


def predict(image_path: str, class_names: list):

    # 1. Load model
    model = create_model(num_classes=config.NUM_CLASSES, freeze_backbone=False)
    model = load_model(model, config.MODEL_DIR / "resnet18_classifier.pth", config.DEVICE)
    model = model.to(config.DEVICE)
    model.eval()

    # 2. Preprocess image — same transforms as val (no augmentation)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(config.DEVICE)  # add batch dim

    # 3. Predict
    with torch.inference_mode():
        logits = model(img_tensor)
        probs  = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()

    print(f"\n[predict] Image      : {image_path}")
    print(f"[predict] Prediction : {class_names[pred_idx]}")
    print(f"[predict] Confidence : {probs[0][pred_idx].item():.4f}")
    print(f"[predict] All probs  :")
    for i, name in enumerate(class_names):
        print(f"            {name:>10} : {probs[0][i].item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    args = parser.parse_args()

    # Must match the order torchvision.datasets.ImageFolder assigned
    CLASS_NAMES = ["bird", "cat", "dog"]

    predict(args.image, CLASS_NAMES)
