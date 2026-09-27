"""Learn two scene facts from the sample videos' tracks and save them to src/assets/scene_priors.npz:

    road   where vehicles drive (the carriageway), as a boolean grid over the reference view
    flow   the usual direction of travel in each cell, plus how consistent it is

    python scripts/build_scene_priors.py samples/*.MP4

Needs cache/<video>.tracks.csv and cache/<video>.signals.json (for the per-video alignment).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.scene import PRIORS_PATH, Scene  # noqa: E402
from src.tracking import load_tracks  # noqa: E402
from src.trajectories import build_tracks  # noqa: E402

CELL = 10                 # grid cell size in reference px
MIN_VEHICLES = 3          # a cell is road if at least this many distinct vehicles drove over it
MOVING = 40.0             # px/s: slower samples say nothing about the direction of travel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    args = ap.parse_args()

    gh, gw = config.REF_H // CELL, config.REF_W // CELL
    visits = np.zeros((gh, gw), np.float32)
    flow = np.zeros((gh, gw, 2), np.float32)
    flow_n = np.zeros((gh, gw), np.float32)
    for video in args.videos:
        name = Path(video).name
        A = np.array(json.loads((config.CACHE_DIR / f"{name}.signals.json").read_text())["align"])
        for tr in build_tracks(load_tracks(config.CACHE_DIR / f"{name}.tracks.csv"), A):
            if not tr.is_vehicle:
                continue
            cells = np.clip((tr.foot / CELL).astype(int), 0, [gw - 1, gh - 1])
            seen = np.zeros((gh, gw), bool)
            for (cx, cy), s, h in zip(cells, tr.speed, tr.heading):
                seen[cy, cx] = True
                if s > MOVING:
                    flow[cy, cx] += h
                    flow_n[cy, cx] += 1
            # the footprint of a vehicle is wider than its centre line: count its neighbourhood once
            visits += cv2.dilate(seen.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        print(f"[{name}] done", flush=True)

    road = (visits >= MIN_VEHICLES).astype(np.uint8)
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # raised islands and the median are never carriageway
    islands = np.zeros((config.REF_H, config.REF_W), np.uint8)
    for poly in Scene.load(with_priors=False).islands.values():
        cv2.fillPoly(islands, [poly.astype(np.int32)], 1)
    road[cv2.resize(islands, (gw, gh), interpolation=cv2.INTER_AREA) > 0.5] = 0

    mean = flow / np.maximum(flow_n[..., None], 1)
    consistency = np.linalg.norm(mean, axis=2)          # 1 = everyone drives the same way here
    direction = mean / np.maximum(consistency[..., None], 1e-6)
    PRIORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRIORS_PATH, cell=CELL, road=road.astype(bool), direction=direction.astype(np.float32),
                        consistency=consistency.astype(np.float32), samples=flow_n.astype(np.int32))
    print(f"road cells: {int(road.sum())} of {gh * gw}; wrote {PRIORS_PATH}")

    # a picture of both priors for checking and for the website
    ref = cv2.imread(str(config.ROOT / "src" / "assets" / "reference.jpg"))
    vis = (ref * 0.45).astype(np.uint8)
    mask = cv2.resize(road * 255, (config.REF_W, config.REF_H), interpolation=cv2.INTER_NEAREST)
    vis[mask > 0] = (vis[mask > 0] * 0.6 + np.array([214, 120, 42]) * 0.4).astype(np.uint8)
    for cy in range(0, gh, 3):
        for cx in range(0, gw, 3):
            if road[cy, cx] and flow_n[cy, cx] >= 5 and consistency[cy, cx] > 0.6:
                c = np.array([cx * CELL + CELL / 2, cy * CELL + CELL / 2])
                cv2.arrowedLine(vis, tuple(c.astype(int)), tuple((c + direction[cy, cx] * 22).astype(int)),
                                (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.35)
    out = config.ROOT / "outputs" / "scene_priors.jpg"
    cv2.imwrite(str(out), vis)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
