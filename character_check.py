import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import config

MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "character_model.pth")
CLASSES_PATH = os.path.join(os.path.dirname(__file__), "data", "character_classes.npy")

_TEMPLATE_SIZE = (128, 128)

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_model = None
_classes = None
_mask = None

# Collapse granular training labels back to display names
_REMAP = {
    "Bowser Jr (Larry)":   "Bowser Jr",
    "Bowser Jr (Roy)":     "Bowser Jr",
    "Bowser Jr (Wendy)":   "Bowser Jr",
    "Bowser Jr (Iggy)":    "Bowser Jr",
    "Bowser Jr (Morton)":  "Bowser Jr",
    "Bowser Jr (Lemmy)":   "Bowser Jr",
    "Bowser Jr (Ludwig)":  "Bowser Jr",
    "Hero (Luminary)":     "Hero",
    "Hero (Erdrick)":      "Hero",
    "Hero (Solo)":         "Hero",
    "Hero (Eight)":        "Hero",
}


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


def _preprocess(bgr):
    resized = cv2.resize(bgr, _TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def load_templates():
    global _model, _classes, _mask
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        raise RuntimeError("Character model not found — run train_character_model.py first")
    _mask = _build_mask()
    _classes = np.load(CLASSES_PATH)
    n_classes = len(_classes)
    print("Loading character model...")
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    _model = model
    print(f"Loaded classifier for {n_classes} characters.")


def identify_character(region_bgr):
    """
    Return (character_name, confidence) where confidence is the softmax probability.
    Returns (None, score) if score is below CONFIDENCE_THRESHOLD.
    """
    rgb = _preprocess(region_bgr)
    tensor = _transform(rgb).unsqueeze(0)
    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    best_idx = int(probs.argmax())
    score = float(probs[best_idx])
    name = _REMAP.get(_classes[best_idx], _classes[best_idx])
    if score >= config.CONFIDENCE_THRESHOLD:
        return name, score
    return None, score
