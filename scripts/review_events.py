"""Contact sheet of every detection of one class: a crop at the middle of each event with the
object that caused it boxed. For checking rules by eye and for the website's class examples.

    python scripts/review_events.py stopped_vehicle samples/*.MP4
Writes outputs/review_<class>.jpg. Uses the cached tracks (scripts/cache_tracks.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.pipeline import analyse  # noqa: E402
from src.rules import detect_all  # noqa: E402
from src.scene import Scene  # noqa: E402

TILE_W, TILE_H = 480, 330
COLS = 3


def tile(video: str, frame_t: float, box_ref: tuple[np.ndarray, float], A: np.ndarray, label: str) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, frame_t * 1000)
    ok, frame = cap.read()
    cap.release()
    frame = cv2.resize(frame, (config.REF_W, config.REF_H))
    foot, h = box_ref
    Ainv = cv2.invertAffineTransform(A)
    x, y = (Ainv[:, :2] @ foot + Ainv[:, 2]).astype(int)
    half = int(max(h * 0.6, 12))
    cv2.rectangle(frame, (x - half, int(y - h)), (x + half, y + 4), (0, 0, 255), 3)
    x0 = int(np.clip(x - 320, 0, config.REF_W - 640))
    y0 = int(np.clip(y - 220, 0, config.REF_H - 440))
    crop = cv2.resize(frame[y0:y0 + 440, x0:x0 + 640], (TILE_W, TILE_H))
    cv2.rectangle(crop, (0, 0), (TILE_W, 30), (0, 0, 0), -1)
    cv2.putText(crop, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return crop


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--max", type=int, default=24, help="at most this many tiles")
    args = ap.parse_args()

    scene = Scene.load()
    tiles = []
    for video in args.videos:
        a = analyse(video, cache_dir=config.CACHE_DIR)
        tracks = {tr.id: tr for tr in a.tracks}
        for start, end, tid in detect_all(a.tracks, scene, a.signal).get(args.label, []):
            tr = tracks[tid]
            mid = (start + end) / 2
            k = int(np.argmin(np.abs(tr.t - mid)))
            label = f"{Path(video).stem} {start:.1f}-{end:.1f}s {tr.type} #{tid}"
            tiles.append(tile(video, mid, (tr.foot[k], float(tr.box_h[k])), a.A, label))
            if len(tiles) >= args.max:
                break
        if len(tiles) >= args.max:
            break
    if not tiles:
        print(f"no {args.label} detections")
        return 0
    while len(tiles) % COLS:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)])
    out = config.ROOT / "outputs" / f"review_{args.label}.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"{len(tiles)} tiles -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
