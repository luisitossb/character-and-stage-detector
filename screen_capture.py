import cv2

import config

_cap = None


def _get_cap():
    global _cap
    if _cap is None or not _cap.isOpened():
        _cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not _cap.isOpened():
            raise RuntimeError(f"Could not open camera index {config.CAMERA_INDEX}")
        _cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        _cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return _cap


def grab_frame():
    """Return the latest frame from the capture card as a BGR numpy array."""
    cap = _get_cap()
    # Flush the buffer by reading a couple frames so we get the freshest image
    for _ in range(2):
        cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        raise RuntimeError("Failed to read frame from capture card")
    return frame


def crop_region(frame, region):
    """Crop a (x, y, w, h) region from frame and return it."""
    x, y, w, h = region
    return frame[y:y + h, x:x + w]


def snapshot(path="snapshot.png"):
    """Save a single frame to disk — useful for calibrating P1/P2 regions."""
    frame = grab_frame()
    cv2.imwrite(path, frame)
    print(f"Saved {path}  ({frame.shape[1]}x{frame.shape[0]})")
    return frame


def release():
    global _cap
    if _cap is not None:
        _cap.release()
        _cap = None


# Run directly to take a calibration snapshot
if __name__ == "__main__":
    snapshot()
