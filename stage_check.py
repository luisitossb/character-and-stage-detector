import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import config

MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "stage_model.pth")
CLASSES_PATH = os.path.join(os.path.dirname(__file__), "data", "stage_classes.npy")

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_model = None
_classes = None


def _preprocess(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_index():
    global _model, _classes
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        raise RuntimeError("Stage model not found — run train_stage_model.py first")
    _classes = np.load(CLASSES_PATH)
    n_classes = len(_classes)
    print("Loading stage model...")
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    _model = model
    print(f"Loaded classifier for {n_classes} stages.")


def identify_stage(frame_bgr):
    """
    Return (stage_name, confidence) where confidence is the softmax probability.
    Returns (None, score) if score is below STAGE_CONFIDENCE_THRESHOLD.
    """
    rgb = _preprocess(frame_bgr)
    tensor = _transform(rgb).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(_model(tensor), dim=1)[0]
    best_idx = int(probs.argmax())
    score = float(probs[best_idx])
    if score >= config.STAGE_CONFIDENCE_THRESHOLD:
        return _classes[best_idx], score
    return None, score
