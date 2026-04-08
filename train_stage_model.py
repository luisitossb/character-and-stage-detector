"""
Fine-tunes MobileNetV3-Small as a stage classifier.
Run once — takes ~15-30 min on CPU.
Saves data/stage_model.pth and data/stage_classes.npy.
"""

import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

STAGES_DIR = os.path.join(os.path.dirname(__file__), "data", "stages")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "stage_model.pth")
CLASSES_PATH = os.path.join(os.path.dirname(__file__), "data", "stage_classes.npy")

_train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _folder_to_name(folder):
    parts = folder.split(" - ", 1)
    return parts[1].strip() if len(parts) == 2 else folder.strip()


def _is_variant(filename):
    """Returns True if this file is a BF or Omega variant — skip these."""
    base = filename.rsplit(".", 1)[0]
    if " - " in base:
        base = base.split(" - ", 1)[1].strip()
    return base.startswith("[BF]") or base.startswith("[\u2127]")


def _load_img(path):
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class StageDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rgb, label = self.items[i]
        return self.transform(rgb), label


def load_data():
    classes = sorted({
        _folder_to_name(f)
        for f in os.listdir(STAGES_DIR)
        if os.path.isdir(os.path.join(STAGES_DIR, f))
    })
    class_to_idx = {c: i for i, c in enumerate(classes)}

    items = []
    for folder in os.listdir(STAGES_DIR):
        folder_path = os.path.join(STAGES_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        name = _folder_to_name(folder)
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(".jpg"):
                continue
            if _is_variant(fname):
                continue  # skip BF and Omega forms
            img = _load_img(os.path.join(folder_path, fname))
            if img is None:
                continue
            items.append((img, class_to_idx[name]))

    return items, classes


def train():
    print("Loading stage images...")
    items, classes = load_data()
    n_classes = len(classes)
    print(f"  {len(items)} images across {n_classes} stages")

    np.save(CLASSES_PATH, np.array(classes))

    random.shuffle(items)
    val_size = max(n_classes, len(items) // 8)
    val_items = items[:val_size]
    train_items = items[val_size:]

    train_ds = StageDataset(train_items, _train_transform)
    val_ds = StageDataset(val_items, _val_transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    print("Building model...")
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val_acc = 0.0
    EPOCHS = 50

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += (model(imgs).argmax(1) == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total else 0.0
        marker = " <-- saved" if val_acc >= best_val_acc else ""
        print(f"  Epoch {epoch+1:2d}/{EPOCHS}  loss={total_loss/len(train_loader):.3f}  val_acc={val_acc:.1%}{marker}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nDone. Best val accuracy: {best_val_acc:.1%}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
