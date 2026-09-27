"""Per-object trajectories in the reference view, with smoothed position, speed and heading."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from . import align
from .tracking import Detection

VEHICLE_TYPES = {"car", "bus", "truck", "motorcycle", "bicycle"}
SMOOTH_SEC = 0.6     # moving-average window for positions
MIN_DETS = 5         # shorter tracks are detector flicker


@dataclass
class Track:
    id: int
    type: str                 # majority class over the track
    t: np.ndarray             # (N,) seconds
    foot: np.ndarray          # (N, 2) bottom-centre of the box = ground contact, reference px (smoothed)
    box_h: np.ndarray         # (N,) box height, reference px (a proxy for distance to the camera)
    speed: np.ndarray         # (N,) px/s
    heading: np.ndarray       # (N, 2) unit direction of motion (0,0 when not moving)

    @property
    def is_vehicle(self) -> bool:
        return self.type in VEHICLE_TYPES

    @property
    def start(self) -> float:
        return float(self.t[0])

    @property
    def end(self) -> float:
        return float(self.t[-1])


def _smooth(values: np.ndarray, t: np.ndarray, window: float) -> np.ndarray:
    """Centered moving average over a time window (tracks are sampled irregularly when boxes drop out)."""
    out = np.empty_like(values)
    lo = hi = 0
    for i in range(len(t)):
        while t[lo] < t[i] - window / 2:
            lo += 1
        while hi + 1 < len(t) and t[hi + 1] <= t[i] + window / 2:
            hi += 1
        out[i] = values[lo:hi + 1].mean(axis=0)
    return out


def build_tracks(dets: list[Detection], A: np.ndarray) -> list[Track]:
    """Group detections by track id, map them into the reference view and derive kinematics."""
    by_id: dict[int, list[Detection]] = defaultdict(list)
    for d in dets:
        by_id[d.track_id].append(d)
    tracks = []
    for tid, ds in by_id.items():
        if len(ds) < MIN_DETS:
            continue
        ds.sort(key=lambda d: d.t)
        t = np.array([d.t for d in ds])
        raw = align.apply_points(A, np.array([((d.x1 + d.x2) / 2, d.y2) for d in ds]))
        top = align.apply_points(A, np.array([((d.x1 + d.x2) / 2, d.y1) for d in ds]))
        foot = _smooth(raw, t, SMOOTH_SEC)
        vel = np.gradient(foot, t, axis=0) if len(t) > 1 else np.zeros_like(foot)
        speed = np.hypot(vel[:, 0], vel[:, 1])
        heading = np.where(speed[:, None] > 1e-6, vel / np.maximum(speed[:, None], 1e-6), 0.0)
        kind = Counter(d.type for d in ds).most_common(1)[0][0]
        tracks.append(Track(tid, kind, t, foot, np.linalg.norm(raw - top, axis=1), speed, heading))
    return tracks


def runs(mask: np.ndarray, t: np.ndarray, min_len: float, max_gap: float = 0.0) -> list[tuple[int, int]]:
    """Index ranges [i, j] where `mask` holds, gaps up to `max_gap` seconds bridged, lasting >= min_len s."""
    out: list[list[int]] = []
    for i in np.flatnonzero(mask):
        if out and t[i] - t[out[-1][1]] <= max(max_gap, 0.35):
            out[-1][1] = i
        else:
            out.append([i, i])
    return [(i, j) for i, j in out if t[j] - t[i] >= min_len]
