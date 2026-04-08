import os

# Capture card camera index
CAMERA_INDEX = 1

# Path to the folder containing per-character subfolders with portrait PNGs
CHARACTER_PNG_DIR = os.path.join(os.path.dirname(__file__), "data", "chars")

# Minimum template match confidence (0.0 - 1.0) to count as a detection
CONFIDENCE_THRESHOLD = 0.83

# How many seconds to wait between detection loops
POLL_INTERVAL = 1.0

# Pixel regions for each player's stock icon on the capture card feed.
# Format: (x, y, width, height)
# Calibrate these by running screen_capture.py in snapshot mode and inspecting
# the saved frame — find the top-left corner and size of each stock icon box.
#
# Assuming 640x480 capture resolution
# Stock icons appear to be roughly 20-25 pixels each, grouped together

# P1 portrait - bottom left, below the damage percentage
P1_REGION = (370, 868, 190, 190)  # x, y, width, height

# P2 portrait - bottom right, below the damage percentage
P2_REGION = (1110, 868, 190, 190)  # x, y, width, height

# Rotated portrait mask — defines the tilted square frame in the 128x128 processed image.
# center: (x, y) of the frame center
# size: (width, height) of the square side length
# angle: degrees counter-clockwise
# Run debug.py to see the mask drawn on your crop and tune these values.
PORTRAIT_MASK_CENTER = (63.5, 67)
PORTRAIT_MASK_SIZE = (85, 85)
PORTRAIT_MASK_ANGLE = -27

# Minimum cosine similarity (0.0 - 1.0) for stage detection
STAGE_CONFIDENCE_THRESHOLD = 0.83
