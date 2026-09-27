"""Shared constants. Every tunable number of the pipeline lives here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
CACHE_DIR = ROOT / "cache"

SEED = 0

# All geometry (scene layout, tracks) is expressed in this reference frame,
# whatever the resolution of the input video.
REF_W, REF_H = 1920, 1080

# Detection / tracking
DETECTOR_WEIGHTS = WEIGHTS_DIR / "yolo11s.pt"
# The live demo runs on a CPU server and overrides these for speed (see demo/Dockerfile).
DETECT_STRIDE = int(os.environ.get("TRAFFIC_DETECT_STRIDE", 3))    # detector on every N-th frame (30 -> 10 fps)
DETECT_IMGSZ = int(os.environ.get("TRAFFIC_DETECT_IMGSZ", 1280))   # detector input size (longest side)
DETECT_CONF = 0.10       # low on purpose: ByteTrack uses low-score boxes in its 2nd association

# COCO class id -> our road-user type
COCO_TO_TYPE = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
