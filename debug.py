"""
Run this while in a match to diagnose detection issues.
Saves p1_crop.png, p1_overlay.png, p2_crop.png, p2_overlay.png.
The overlay shows the mask region — tune PORTRAIT_MASK_SIZE in config.py to adjust it.
"""

import cv2
import numpy as np

import character_check
import config
import screen_capture
import stage_check

_TEMPLATE_SIZE = (128, 128)


def _build_mask_corners():
    cx, cy = config.PORTRAIT_MASK_CENTER
    rw, rh = config.PORTRAIT_MASK_SIZE[0] / 2, config.PORTRAIT_MASK_SIZE[1] / 2
    angle = np.radians(config.PORTRAIT_MASK_ANGLE)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    corners = np.array([[-rw, -rh], [rw, -rh], [rw, rh], [-rw, rh]], dtype=np.float32)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    return (corners @ R.T + np.array([cx, cy])).astype(np.int32)


_MASK_CORNERS = _build_mask_corners()


def top_character_matches(region_bgr, n=5):
    import torch
    rgb = character_check._preprocess(region_bgr)
    tensor = character_check._transform(rgb).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(character_check._model(tensor), dim=1)[0]
    top_idx = probs.argsort(descending=True)[:n]
    return [(character_check._classes[i], float(probs[i])) for i in top_idx]


def top_stage_matches(frame_bgr, n=5):
    import torch
    rgb = stage_check._preprocess(frame_bgr)
    tensor = stage_check._transform(rgb).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(stage_check._model(tensor), dim=1)[0]
    top_idx = probs.argsort(descending=True)[:n]
    return [(stage_check._classes[i], float(probs[i])) for i in top_idx]


def main():
    print("Loading character index...")
    character_check.load_templates()
    stage_check.load_index()
    print()

    frame = screen_capture.grab_frame()
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}\n")

    # Stage
    stage_matches = top_stage_matches(frame)
    print("Stage top matches:")
    for name, score in stage_matches:
        bar = "#" * max(0, int(score * 40))
        print(f"  {score:.3f}  {bar}  {name}")
    print()

    # Characters
    for player, region in [("P1", config.P1_REGION), ("P2", config.P2_REGION)]:
        crop = screen_capture.crop_region(frame, region)
        cv2.imwrite(f"{player.lower()}_crop.png", crop)

        resized = cv2.resize(crop, _TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
        overlay = resized.copy()
        cv2.polylines(overlay, [_MASK_CORNERS], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imwrite(f"{player.lower()}_overlay.png", overlay)

        matches = top_character_matches(crop)

        print(f"{player} region {region} -> {crop.shape[1]}x{crop.shape[0]}")
        print(f"  Saved: {player.lower()}_crop.png  {player.lower()}_overlay.png")
        print("  Top matches:")
        for name, score in matches:
            bar = "#" * max(0, int(score * 40))
            print(f"    {score:.3f}  {bar}  {name}")
        print()

    screen_capture.release()


if __name__ == "__main__":
    main()
