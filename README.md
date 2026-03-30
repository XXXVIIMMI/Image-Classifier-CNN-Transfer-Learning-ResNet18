# Image Classifier — CNN + Transfer Learning (ResNet18)

A modular PyTorch image classifier using transfer learning with ResNet18.
Classifies images into 3 categories: **bird, cat, dog**.

---

## Project Structure

```
image_classifier/
│
├── src/
│   ├── __init__.py         ← makes src/ a Python package
│   ├── config.py           ← all settings and hyperparameters
│   ├── download_data.py    ← downloads and prepares dataset
│   ├── data_setup.py       ← creates DataLoaders
│   ├── model.py            ← defines ResNet18 model architecture
│   ├── engine.py           ← train and eval loops
│   └── utils.py            ← save/load model, plot results
│
├── data/
│   ├── raw/                ← original CIFAR-10 download (auto-created)
│   ├── train/              ← training image folders (auto-created)
│   └── val/                ← validation image folders (auto-created)
│
├── models/                 ← saved model weights (auto-created)
├── pic/                    ← sample test images (user-provided)
├── train.py                ← entrypoint to train model
├── predict.py              ← entrypoint to predict one image
├── requirements.txt       ← Python dependencies
└── README.md              ← project documentation
```

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/XXXVIIMMI/Image-Classifier-CNN-Transfer-Learning-ResNet18.git
cd Image-Classifier-CNN-Transfer-Learning-ResNet18
```

### 2. Create conda environment

```bash
conda create -n img_clf python=3.11
conda activate img_clf
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train

```bash
python train.py
```

That's it. On first run it will:

1. Download CIFAR-10 automatically (no login needed)
2. Extract only bird, cat, dog images
3. Train ResNet18 for 10 epochs
4. Save model to `models/resnet18_classifier.pth`
5. Save loss/accuracy plot to `results.png`

---

## Predict on a New Image

```bash
python predict.py --image path/to/your/image.jpg
example: python predict.py --image pic/dog.jpg
```

Example output:

```
[predict] Image      : my_cat.jpg
[predict] Prediction : cat
[predict] Confidence : 0.9231
[predict] All probs  :
              bird : 0.0312
               cat : 0.9231
               dog : 0.0457
```

---

## Configuration

All settings are in `src/config.py`. Change values there only.

| Setting       | Default | Description               |
| ------------- | ------- | ------------------------- |
| NUM_CLASSES   | 3       | Number of categories      |
| BATCH_SIZE    | 32      | Images per batch          |
| NUM_EPOCHS    | 10      | Training epochs           |
| LEARNING_RATE | 0.0001  | Adam optimizer LR         |
| IMG_SIZE      | 224     | Input image size (pixels) |

---

## Model Details

| Item          | Detail                                                                       |
| ------------- | ---------------------------------------------------------------------------- |
| Backbone      | ResNet18 pretrained on ImageNet                                              |
| Head          | Linear(512 → 256) → BatchNorm1d(256) → ReLU → Dropout(0.4) → Linear(256 → 3) |
| Frozen layers | All backbone layers (head trained only when freeze_backbone=True)            |
| Optimizer     | Adam                                                                         |
| Loss          | CrossEntropyLoss                                                             |

---

## File Responsibilities

| File             | One job                                  |
| ---------------- | ---------------------------------------- |
| config.py        | Every number and path lives here         |
| download_data.py | Download dataset, build folder structure |
| data_setup.py    | Give me a folder → return DataLoaders    |
| model.py         | Give me num_classes → return model       |
| engine.py        | Give me model + data → run one epoch     |
| utils.py         | Save, load, and plot                     |
| train.py         | Calls all of the above in order          |
| predict.py       | Load saved model, predict one image      |

---

## Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA optional (CPU works fine for this dataset size)
