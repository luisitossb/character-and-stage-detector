import time

import pyautogui

# Pause between pyautogui actions (seconds)
pyautogui.PAUSE = 0.05


def type_characters(p1: str | None, p2: str | None, stage: str | None = None):
    """
    Type 'P1Name vs P2Name on Stage' into whatever browser field currently has focus.
    Call this only when the detected characters or stage have changed.
    """
    p1_str = p1 if p1 else "Unknown"
    p2_str = p2 if p2 else "Unknown"
    output = f"{p1_str} vs {p2_str}"
    if stage:
        output += f" on {stage}"

    # Select all existing text and replace it
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.05)
    pyautogui.typewrite(output, interval=0.03)
