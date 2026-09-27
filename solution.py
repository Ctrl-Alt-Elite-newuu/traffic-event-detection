"""
solution.py — entry point called by the organizers' harness (run_submission.py).

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B

Part A: one pass over the video (camera alignment, YOLO11s + ByteTrack, traffic-light reading;
src/pipeline.py), then per-class rules on the trajectories (src/rules.py) and segment clean-up
(src/segments.py). Part B: a causal time-to-collision estimate on tracks built from the frames
seen so far (src/risk.py); it never opens the video and never uses Part A's output.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# Official class ids (14). Remove entries you never predict; never add.
CLASSES: list[str] = [
    "accident",            # collision between road users / with a fixed object
    "near_miss",           # sharp braking or swerving to avoid a collision, no contact
    "red_light",           # crossing the stop line on red
    "wrong_way",           # driving against the traffic direction / in the oncoming lane
    "illegal_u_turn",      # U-turn where prohibited
    "stopped_vehicle",     # stationary on the carriageway >= 10 s, not queued at a signal
    "jaywalking",          # pedestrian on the carriageway outside a crossing
    "failure_to_yield",    # driving through a crossing while a pedestrian is on it
    "illegal_turn",        # turn from the wrong lane or in a prohibited direction
    "solid_line_crossing", # lane change / manoeuvre across a solid marking
    "stop_line",           # stopped past the stop line on red
    "congestion",          # standstill / crawling traffic across all lanes of a direction
    "road_obstacle",       # debris, animal or fallen object on the carriageway
    "fire_smoke",          # visible fire or smoke from a vehicle or on the road
]

# Classes our rules predict. Scored classes are the union of the test set's classes and the ones we
# predict, so a class is only enabled when its rule is precise enough to be worth the risk.
ENABLED: list[str] = ["jaywalking", "red_light", "stopped_vehicle", "wrong_way", "failure_to_yield"]

# Horizon used by the metric (seconds).
RISK_HORIZON_SEC = 5.0

# Optional: reuse cached tracks/signals (development only; the official run leaves this unset).
CACHE_DIR = os.environ.get("TRAFFIC_CACHE_DIR")


def _seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def detect_events(video_path: str) -> list[list]:
    """Part A — [[start_sec, end_sec, label], ...] for one .mp4."""
    from src.pipeline import analyse
    from src.rules import detect_all
    from src.scene import Scene
    from src.segments import to_events

    _seed()
    analysis = analyse(video_path, cache_dir=Path(CACHE_DIR) if CACHE_DIR else None)
    by_class = detect_all(analysis.tracks, Scene.load(), analysis.signal)
    return to_events(by_class, analysis.info.duration, ENABLED)


class RiskEstimator:
    """Part B — causal accident anticipation: P(accident starts within RISK_HORIZON_SEC)."""

    def reset(self, meta: dict) -> None:
        from src.risk import RiskModel

        _seed()
        self.model = RiskModel(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return self.model.step(frame, t_sec)
