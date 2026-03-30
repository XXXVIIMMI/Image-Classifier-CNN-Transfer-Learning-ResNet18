import torch
from pathlib import Path

# --- Paths ---
DATA_DIR  = Path("data/")
MODEL_DIR = Path("models/")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- Training settings ---
BATCH_SIZE    = 32
NUM_EPOCHS    = 10
LEARNING_RATE = 0.0001
NUM_WORKERS   = 2

# --- Model settings ---
IMG_SIZE    = 224       # ResNet expects 224x224
NUM_CLASSES = 3         # bird, cat, dog

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[config] Using device: {DEVICE}")
