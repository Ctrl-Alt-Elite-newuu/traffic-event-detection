"""Export EDA data and figures for the team website from the cached tracks and signal phases.

    python scripts/export_site_data.py samples/*.MP4 --out ../traffic-event-detection-web/public/data

Writes:
    videos.json               per-video facts, counts over time, track totals, signal phases, alignment
    eda/*.jpg                 thumbnails, scene layout, motion heatmaps, trajectories, alignment check
    results/<video>.json      events and risk curve per video, with --predictions predictions_samples.json
Needs cache/<video>.tracks.csv (scripts/cache_tracks.py) and, for phases, cache/<video>.signals.json
(scripts/cache_signals.py).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import align, config  # noqa: E402
from src.scene import Scene  # noqa: E402
from src.tracking import load_tracks  # noqa: E402
from src.video import iter_frames, probe  # noqa: E402

BIN_SEC = 5.0
MIN_CONF = 0.3
MIN_TRACK_DETS = 10          # ~1 s at the 10 Hz detection rate: shorter tracks are fragments
GROUPS = {"car": "cars", "bus": "heavy", "truck": "heavy", "motorcycle": "two_wheelers",
          "bicycle": "two_wheelers", "person": "pedestrians"}
# Sequential ramps from the site's chart palette (light -> dark).
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]
ORANGE_RAMP = ["#fde3d6", "#f5ad8c", "#eb6834", "#c24d1c", "#8a3310"]
JPEG = [cv2.IMWRITE_JPEG_QUALITY, 82]


def hex_to_bgr(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)], dtype=np.float32)


def ramp_lut(ramp: list[str]) -> np.ndarray:
    """256-entry BGR lookup table interpolating a sequential ramp."""
    stops = np.stack([hex_to_bgr(c) for c in ramp])
    x = np.linspace(0, 1, len(stops))
    t = np.linspace(0, 1, 256)
    return np.stack([np.interp(t, x, stops[:, i]) for i in range(3)], axis=1)


def heat_overlay(bg: np.ndarray, paths: list[np.ndarray], ramp: list[str]) -> np.ndarray:
    """Flow heatmap: how many distinct road users passed through each spot (not how long they stood there)."""
    h, w = bg.shape[:2]
    s = 4
    hist = np.zeros((h // s, w // s), np.float32)
    for path in paths:
        mask = np.zeros_like(hist, np.uint8)
        cv2.polylines(mask, [(path / s).astype(np.int32).reshape(-1, 1, 2)], False, 1, thickness=3)
        hist += mask
    hist = cv2.GaussianBlur(hist, (0, 0), 2)
    hist = np.clip(hist / max(np.percentile(hist[hist > 0], 99) if (hist > 0).any() else 1, 1e-6), 0, 1)
    hist = cv2.resize(hist, (w, h), interpolation=cv2.INTER_LINEAR)
    lut = ramp_lut(ramp)
    color = lut[(hist * 255).astype(np.uint8)]
    alpha = np.clip(hist * 1.4, 0, 0.92)[..., None]
    base = (cv2.cvtColor(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) * 0.5).astype(np.float32)
    return (base * (1 - alpha) + color * alpha).astype(np.uint8)


def heading_color(dx: float, dy: float) -> tuple[int, int, int]:
    hue = int((math.degrees(math.atan2(dy, dx)) % 360) / 2)
    b, g, r = cv2.cvtColor(np.uint8([[[hue, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def first_frame(video: str) -> np.ndarray:
    return next(iter_frames(video))[2]


def export_video(video: str, out_dir: Path, all_pts: dict, traj_canvas: np.ndarray) -> dict:
    name = Path(video).name
    vid = Path(video).stem
    info = probe(video)
    dets = [d for d in load_tracks(config.CACHE_DIR / f"{name}.tracks.csv") if d.conf >= MIN_CONF]
    sig_path = config.CACHE_DIR / f"{name}.signals.json"
    signals = json.loads(sig_path.read_text()) if sig_path.exists() else None
    A = np.array(signals["align"]) if signals else align.estimate_for_video(video, info.fps)[0]

    frame = first_frame(video)
    thumb = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_dir / "eda" / f"thumb_{vid}.jpg"), thumb, JPEG)
    brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2].mean())

    # counts over time: mean number of visible road users per detector frame, per group
    n_bins = int(math.ceil(info.duration / BIN_SEC))
    frames_in_bin = [set() for _ in range(n_bins)]
    sums = {g: np.zeros(n_bins) for g in set(GROUPS.values())}
    for d in dets:
        b = min(int(d.t // BIN_SEC), n_bins - 1)
        frames_in_bin[b].add(d.frame)
        sums[GROUPS[d.type]][b] += 1
    denom = np.array([max(len(f), 1) for f in frames_in_bin])
    counts = {"t": [round(i * BIN_SEC, 1) for i in range(n_bins)]}
    counts.update({g: np.round(v / denom, 2).tolist() for g, v in sums.items()})

    # unique road users (tracks long enough to be real), and trajectories in the reference view
    by_track: dict[int, list] = defaultdict(list)
    for d in dets:
        by_track[d.track_id].append(d)
    totals: Counter = Counter()
    for tid, ds in by_track.items():
        if len(ds) < MIN_TRACK_DETS:
            continue
        kind = Counter(d.type for d in ds).most_common(1)[0][0]
        totals[GROUPS[kind]] += 1
        foot = align.apply_points(A, np.array([((d.x1 + d.x2) / 2, d.y2) for d in ds]))
        key = "pedestrians" if kind == "person" else "vehicles"
        all_pts[key].append(foot)
        if kind != "person" and np.linalg.norm(foot[-1] - foot[0]) > 60:
            for p, q in zip(foot[:-1], foot[1:]):
                cv2.line(traj_canvas, tuple(p.astype(int)), tuple(q.astype(int)),
                         heading_color(*(q - p)), 1, cv2.LINE_AA)

    s = math.hypot(A[0][0], A[1][0])
    return {
        "id": vid,
        "file": name,
        "duration": round(info.duration, 2),
        "fps": round(info.fps, 3),
        "width": info.width,
        "height": info.height,
        "n_frames": info.n_frames,
        "size_gb": round(Path(video).stat().st_size / 1e9, 2),
        "bitrate_mbps": round(Path(video).stat().st_size * 8 / info.duration / 1e6, 1),
        "brightness": round(brightness, 1),
        "lighting": "daylight" if brightness >= 95 else "dusk",
        "thumb": f"eda/thumb_{vid}.jpg",
        "align": {"dx": round(A[0][2], 1), "dy": round(A[1][2], 1), "scale": round(s, 4),
                  "rot_deg": round(math.degrees(math.atan2(A[1][0], A[0][0])), 2),
                  "inliers": signals["align_inliers"] if signals else None},
        "tracks": dict(totals),
        "counts": counts,
        "signals": signals["phases"] if signals else [],
    }


def alignment_figure(video: str, out: Path) -> None:
    """Side by side: raw frame vs aligned frame, each blended with the reference view."""
    ref = cv2.imread(str(align.REFERENCE_IMAGE))
    f = first_frame(video)
    A, _ = align.estimate(f)
    warped = cv2.warpAffine(f, A, (config.REF_W, config.REF_H))
    crop = (slice(260, 700), slice(760, 1540))
    raw = cv2.addWeighted(f, 0.5, ref, 0.5, 0)[crop]
    fixed = cv2.addWeighted(warped, 0.5, ref, 0.5, 0)[crop]
    cv2.imwrite(str(out), np.hstack([raw, np.full((raw.shape[0], 8, 3), 255, np.uint8), fixed]), JPEG)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, help="predictions.json from run_submission.py, for results/")
    args = ap.parse_args()
    (args.out / "eda").mkdir(parents=True, exist_ok=True)

    ref = cv2.imread(str(align.REFERENCE_IMAGE))
    all_pts: dict[str, list] = {"vehicles": [], "pedestrians": []}
    traj = (ref * 0.35).astype(np.uint8)
    videos = []
    for v in args.videos:
        videos.append(export_video(v, args.out, all_pts, traj))
        print(f"[{Path(v).name}] exported", flush=True)

    for key, ramp in (("vehicles", BLUE_RAMP), ("pedestrians", ORANGE_RAMP)):
        cv2.imwrite(str(args.out / "eda" / f"heat_{key}.jpg"), heat_overlay(ref, all_pts[key], ramp), JPEG)
    cv2.imwrite(str(args.out / "eda" / "trajectories.jpg"), traj, JPEG)
    cv2.imwrite(str(args.out / "eda" / "scene_layout.jpg"), Scene.load().draw(ref), JPEG)
    shifted = max(videos, key=lambda d: abs(d["align"]["dx"]) + abs(d["align"]["dy"]))
    alignment_figure(next(v for v in args.videos if Path(v).stem == shifted["id"]),
                     args.out / "eda" / "alignment.jpg")

    (args.out / "videos.json").write_text(json.dumps({"videos": videos, "alignment_example": shifted["id"]},
                                                     separators=(",", ":")))
    if args.predictions:
        (args.out / "results").mkdir(exist_ok=True)
        for name, pred in json.loads(args.predictions.read_text())["videos"].items():
            risk = pred["risk"][::6]                    # ~5 points per second is plenty for a chart
            result = {"events": pred["events"], "risk": risk}
            if (args.out / "results" / f"{Path(name).stem}.mp4").exists():   # from render_annotated.py
                result["annotated_video"] = f"results/{Path(name).stem}.mp4"
            (args.out / "results" / f"{Path(name).stem}.json").write_text(json.dumps(result, separators=(",", ":")))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
