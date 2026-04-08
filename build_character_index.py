"""
Run this once to fingerprint all chara_4 portrait PNGs.
Saves data/character_index.npy and data/character_names.npy.
Re-run only if you add new characters to data/chars/.
"""

import os
import re

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms

import config

CHARS_DIR = config.CHARACTER_PNG_DIR
INDEX_PATH = os.path.join(os.path.dirname(__file__), "data", "character_index.npy")
NAMES_PATH = os.path.join(os.path.dirname(__file__), "data", "character_names.npy")

_PORTRAIT_RE = re.compile(r"^chara_4_.+_0[0-7]\.png$", re.IGNORECASE)
_TEMPLATE_SIZE = (128, 128)

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _build_mask():
    h, w = _TEMPLATE_SIZE[1], _TEMPLATE_SIZE[0]
    cx, cy = config.PORTRAIT_MASK_CENTER
    rw, rh = config.PORTRAIT_MASK_SIZE[0] / 2, config.PORTRAIT_MASK_SIZE[1] / 2
    angle = np.radians(config.PORTRAIT_MASK_ANGLE)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    corners = np.array([[-rw, -rh], [rw, -rh], [rw, rh], [-rw, rh]], dtype=np.float32)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    rotated = (corners @ R.T + np.array([cx, cy])).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [rotated], 255)
    return mask


_MASK = _build_mask()


def _build_extractor():
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    extractor = torch.nn.Sequential(model.features, model.avgpool)
    extractor.eval()
    return extractor


def _extract(extractor, bgr):
    resized = cv2.resize(bgr, _TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
    resized[_MASK == 0] = 0
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = _transform(rgb).unsqueeze(0)
    with torch.no_grad():
        feat = extractor(tensor)
    return feat.squeeze().numpy()


def _folder_to_name(folder):
    for sep in (" - ", " . "):
        parts = folder.split(sep, 1)
        if len(parts) == 2:
            return parts[1].strip()
    return folder.strip()


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


def build():
    print("Loading MobileNet...")
    extractor = _build_extractor()

    features = []
    names = []

    for folder in sorted(os.listdir(CHARS_DIR)):
        folder_path = os.path.join(CHARS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        name = _folder_to_name(folder)
        count = 0
        for fname in os.listdir(folder_path):
            if not _PORTRAIT_RE.match(fname):
                continue
            img = _load_img(os.path.join(folder_path, fname))
            if img is None:
                continue
            feat = _extract(extractor, img)
            features.append(feat)
            names.append(name)
            count += 1
        if count:
            print(f"  {name} ({count} portraits)")

    np.save(INDEX_PATH, np.array(features, dtype=np.float32))
    np.save(NAMES_PATH, np.array(names))
    print(f"\nDone — {len(features)} fingerprints across {len(set(names))} characters.")


if __name__ == "__main__":
    build()
