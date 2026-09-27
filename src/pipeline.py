"""One pass over a video: camera alignment, detection + tracking, and traffic-light reading.

The 4K stream is decoded exactly once (every DETECT_STRIDE-th frame is converted, the rest are
only grabbed). Everything downstream - the event rules - works on the result, not on pixels.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import align, config
from .scene import Scene
from .signals import SignalTimeline, light_state
from .tracking import Detection, Tracker, load_tracks, save_tracks
from .trajectories import Track, build_tracks
from .video import VideoInfo, iter_frames, probe

SIGNAL_HZ = 2.0
ALIGN_PROBE_SEC = 4.0     # alignment is estimated on frames this far apart; the best of the first few wins


@dataclass
class Analysis:
    info: VideoInfo
    A: np.ndarray                 # video -> reference similarity transform
    tracks: list[Track]           # in reference coordinates
    signal: SignalTimeline


def _analyse(path: str, info: VideoInfo, on_progress: Callable[[float], None] | None
             ) -> tuple[np.ndarray, int, list[Detection], list[tuple[float, str]]]:
    tracker = Tracker()
    signal_every = max(1, round(info.fps / SIGNAL_HZ))
    probe_every = max(1, round(info.fps * ALIGN_PROBE_SEC))
    A, inliers, lamps = np.eye(2, 3), 0, None
    dets: list[Detection] = []
    samples: list[tuple[float, str]] = []
    for idx, t, frame in iter_frames(path, config.DETECT_STRIDE):
        # same frames and criterion as align.estimate_for_video, without decoding them twice
        if idx % probe_every == 0 and (idx // probe_every < align.N_PROBE_FRAMES or lamps is None):
            A_try, n = align.estimate(frame)
            if n >= align.MIN_INLIERS and n > inliers:
                A, inliers = A_try, n
                lamps = Scene.load(with_priors=False).to_video(A).lights["main"]
        dets += tracker.update(frame, idx, t)
        if lamps is not None and idx % signal_every == 0:
            samples.append((t, light_state(frame, lamps)))
            if on_progress and info.duration:
                on_progress(min(1.0, t / info.duration))
    return A, inliers, dets, samples


def analyse(path: str, cache_dir: Path | None = None,
            on_progress: Callable[[float], None] | None = None) -> Analysis:
    """Run (or, with cache_dir, reuse) the pass over one video; on_progress gets the fraction done."""
    info = probe(path)
    name = Path(path).name
    tracks_file = cache_dir / f"{name}.tracks.csv" if cache_dir else None
    signal_file = cache_dir / f"{name}.signals.json" if cache_dir else None
    if tracks_file and signal_file and tracks_file.exists() and signal_file.exists():
        cached = json.loads(signal_file.read_text())
        A = np.array(cached["align"])
        dets = load_tracks(tracks_file)
        signal = SignalTimeline(cached["phases"])
    else:
        A, inliers, dets, samples = _analyse(path, info, on_progress)
        signal = SignalTimeline.from_samples(samples, info.duration)
        if cache_dir:
            save_tracks(dets, tracks_file)
            signal_file.write_text(json.dumps({"video": name, "align": A.round(5).tolist(), "align_inliers": inliers,
                                               "phases": signal.phases}, indent=1))
    return Analysis(info, A, build_tracks(dets, A), signal)
