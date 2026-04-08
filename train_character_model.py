"""
Fine-tunes MobileNetV3-Small as a character classifier.
Uses all chara_ types for richer training data.
Bowser Jr koopalings and Hero variants are split into sub-classes during
training then remapped back to "Bowser Jr" / "Hero" at inference time.
Run once — saves data/character_model.pth and data/character_classes.npy.
"""

import os
import random
import re

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import config

CHARS_DIR = config.CHARACTER_PNG_DIR
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "character_model.pth")
CLASSES_PATH = os.path.join(os.path.dirname(__file__), "data", "character_classes.npy")

# Match all chara types (0-7), all alts (00-07)
_PNG_RE = re.compile(r"^chara_0_.+_0[0-7]\.png$", re.IGNORECASE)

# Koopaling alt index → training label
_KOOPAJR_LABELS = {
    0: "Bowser Jr",
    1: "Bowser Jr (Larry)",
    2: "Bowser Jr (Roy)",
    3: "Bowser Jr (Wendy)",
    4: "Bowser Jr (Iggy)",
    5: "Bowser Jr (Morton)",
    6: "Bowser Jr (Lemmy)",
    7: "Bowser Jr (Ludwig)",
}

# Hero alt index → training label (alts 4-7 repeat 0-3)
_HERO_LABELS = {
    0: "Hero (Luminary)", 1: "Hero (Erdrick)",
    2: "Hero (Solo)",     3: "Hero (Eight)",
    4: "Hero (Luminary)", 5: "Hero (Erdrick)",
    6: "Hero (Solo)",     7: "Hero (Eight)",
}

_train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.5, hue=0.05),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
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
    for sep in (" - ", " . "):
        parts = folder.split(sep, 1)
        if len(parts) == 2:
            return parts[1].strip()
    return folder.strip()


def _file_label(folder_name, filename):
    """Get the training label for a file, handling Koopaling/Hero splits."""
    base = filename.rsplit(".", 1)[0]       # strip .png
    parts = base.split("_")                 # chara, X, name..., alt
    alt = int(parts[-1])
    char_id = "_".join(parts[2:-1])         # internal name e.g. "koopajr", "brave"
    if char_id == "koopajr":
        return _KOOPAJR_LABELS[alt]
    if char_id == "brave":
        return _HERO_LABELS[alt]
    return _folder_to_name(folder_name)


def _load_img(path):
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3].copy()
        bgr[img[:, :, 3] == 0] = 0
        return bgr
    return img[:, :, :3]


def _preprocess(bgr):
    resized = cv2.resize(bgr, (128, 128), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


class PortraitDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rgb, label = self.items[i]
        return self.transform(rgb), label


def load_data():
    # Collect all labels first to build class list
    all_labels = set()
    folder_files = []
    for folder in os.listdir(CHARS_DIR):
        folder_path = os.path.join(CHARS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not _PNG_RE.match(fname):
                continue
            label = _file_label(folder, fname)
            all_labels.add(label)
            folder_files.append((folder_path, fname, label))

    classes = sorted(all_labels)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    items = []
    for folder_path, fname, label in folder_files:
        img = _load_img(os.path.join(folder_path, fname))
        if img is None:
            continue
        items.append((_preprocess(img), class_to_idx[label]))

    return items, classes


def train():
    print("Loading portraits...")
    items, classes = load_data()
    n_classes = len(classes)
    print(f"  {len(items)} images across {n_classes} classes")

    np.save(CLASSES_PATH, np.array(classes))

    random.shuffle(items)
    val_size = max(n_classes, len(items) // 8)
    val_items = items[:val_size]
    train_items = items[val_size:]

    train_ds = PortraitDataset(train_items, _train_transform)
    val_ds = PortraitDataset(val_items, _val_transform)
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
