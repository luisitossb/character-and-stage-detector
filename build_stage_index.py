"""
Run this once to fingerprint all stage JPGs.
Saves data/stage_index.npy and data/stage_names.npy.
Re-run only if you add new stages to data/stages/.
"""

import os

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms

STAGES_DIR = os.path.join(os.path.dirname(__file__), "data", "stages")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "data", "stage_index.npy")
NAMES_PATH = os.path.join(os.path.dirname(__file__), "data", "stage_names.npy")

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _build_extractor():
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    extractor = torch.nn.Sequential(model.features, model.avgpool)
    extractor.eval()
    return extractor


def _extract(extractor, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = _transform(rgb).unsqueeze(0)
    with torch.no_grad():
        feat = extractor(tensor)
    return feat.squeeze().numpy()


def _folder_to_name(folder):
    parts = folder.split(" - ", 1)
    return parts[1].strip() if len(parts) == 2 else folder.strip()


def build():
    print("Loading MobileNet...")
    extractor = _build_extractor()

    features = []
    names = []

    folders = sorted(os.listdir(STAGES_DIR))
    for folder in folders:
        folder_path = os.path.join(STAGES_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        stage_name = _folder_to_name(folder)
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(".jpg"):
                continue
            buf = np.fromfile(os.path.join(folder_path, fname), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                continue
            feat = _extract(extractor, img)
            features.append(feat)
            names.append(stage_name)
        print(f"  {stage_name} ({len([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')])} images)")

    np.save(INDEX_PATH, np.array(features, dtype=np.float32))
    np.save(NAMES_PATH, np.array(names))
    print(f"\nDone — {len(features)} fingerprints across {len(set(names))} stages.")


if __name__ == "__main__":
    build()
