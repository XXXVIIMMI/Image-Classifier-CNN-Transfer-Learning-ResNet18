import torch.nn as nn
from torchvision import models


def create_model(num_classes: int, freeze_backbone: bool = True):

    # 1. Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 2. Freeze all backbone layers
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # 3. Replace the final FC layer with our custom head
    in_features = model.fc.in_features   # 512

    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(256, num_classes)
    )

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Total params     : {total_params:,}")
    print(f"[model] Trainable params : {trainable_params:,}")

    return model
