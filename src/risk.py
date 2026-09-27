"""Part B: causal accident risk from time-to-collision between tracked road users.

RiskModel sees frames one by one (it never opens the video). Every RISK_STRIDE-th frame it runs the
detector + tracker, maps ground-contact points into the reference view, and RiskCore scores every
pair of road users by how soon, and how hard, they would meet if both kept their current velocity.

Tuned on the sample videos (18 min of normal traffic, no accidents) to about 0.1 false alarms per
minute. Two things caused almost all false alarms and are excluded: a moving car passing a queued
one in the next lane (in this oblique view neighbouring lanes are only pixels apart), and pedestrians
walking past cars at the crossings. So only vehicle pairs that are both moving, on converging
headings, closing fast and about to touch within MAX_TTC_SEC count.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from . import align
from .tracking import Tracker
from .trajectories import Track
from .video import to_ref

RISK_HZ = 5.0
RISK_IMGSZ = 960
HISTORY_SEC = 1.2          # velocity is fitted over this window
MIN_HISTORY_SEC = 0.6
MAX_TTC_SEC = 1.5          # conflicts further away than this are ordinary traffic at a junction
CONTACT = 0.35             # contact radius as a fraction of the two boxes' summed heights
MIN_MOVING = 0.6           # box heights per second: the slower vehicle must be moving too
MIN_CLOSING = 2.0          # box heights per second of closing speed
MIN_ANGLE_DEG = 25.0       # same-direction pairs are one behind the other in a lane
TTC_SCALE = 0.8            # score = exp(-ttc / TTC_SCALE) * closing factor
DECAY = 0.7                # per update; keeps an alarm up for ~1 s after the conflict clears


class RiskCore:
    """Pure scoring part, fed with reference-view observations (also used offline for calibration)."""

    def __init__(self) -> None:
        self.hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=int(HISTORY_SEC * RISK_HZ) + 2))
        self.score = 0.0

    def update(self, t: float, obs: list[tuple[int, bool, np.ndarray, float]]) -> float:
        """obs: (track_id, is_person, foot_xy, box_height) for everything visible at time t."""
        seen = set()
        for tid, is_person, foot, h in obs:
            self.hist[tid].append((t, float(foot[0]), float(foot[1]), float(h), is_person))
            seen.add(tid)
        for tid in [k for k, q in self.hist.items() if q[-1][0] < t - HISTORY_SEC]:
            del self.hist[tid]
        movers = []
        for tid in seen:
            q = [o for o in self.hist[tid] if o[0] >= t - HISTORY_SEC]
            if len(q) < 3 or q[-1][0] - q[0][0] < MIN_HISTORY_SEC:
                continue
            arr = np.array([o[:4] for o in q])
            ts = arr[:, 0] - arr[-1, 0]
            vel = np.array([np.polyfit(ts, arr[:, k], 1)[0] for k in (1, 2)])
            movers.append((arr[-1, 1:3], vel, float(np.median(arr[:, 3])), q[-1][4]))
        raw = 0.0
        for a in range(len(movers)):
            for b in range(a + 1, len(movers)):
                raw = max(raw, self._pair(movers[a], movers[b]))
        self.score = max(raw, self.score * DECAY)
        return self.score

    @staticmethod
    def _pair(m1, m2) -> float:
        (p1, v1, h1, person1), (p2, v2, h2, person2) = m1, m2
        if person1 or person2:
            return 0.0
        s1, s2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if min(s1 / h1, s2 / h2) < MIN_MOVING:
            return 0.0
        if np.degrees(np.arccos(np.clip(v1 @ v2 / (s1 * s2), -1, 1))) < MIN_ANGLE_DEG:
            return 0.0
        p, v = p2 - p1, v2 - v1
        vv = float(v @ v)
        if vv < 1e-6:
            return 0.0
        ttc = -float(p @ v) / vv
        if not 0.0 < ttc < MAX_TTC_SEC or np.linalg.norm(p + v * ttc) > CONTACT * (h1 + h2):
            return 0.0
        closing = -float(p @ v) / max(float(np.linalg.norm(p)), 1e-6) / ((h1 + h2) / 2)
        if closing < MIN_CLOSING:
            return 0.0
        return float(np.exp(-ttc / TTC_SCALE) * min(1.0, closing / 3.0))


class RiskModel:
    """Frame-by-frame wrapper: detection + tracking at RISK_HZ, alignment from the first frames."""

    def __init__(self, meta: dict):
        self.stride = max(1, round(float(meta.get("fps") or 25.0) / RISK_HZ))
        self.tracker = Tracker(imgsz=RISK_IMGSZ)
        self.core = RiskCore()
        self.A: np.ndarray | None = None
        self.align_tries = 0
        self.idx = -1

    def step(self, frame: np.ndarray, t: float) -> float:
        self.idx += 1
        if self.idx % self.stride:
            return self.core.score
        small = to_ref(frame)
        if self.A is None and self.align_tries < align.N_PROBE_FRAMES and self.idx % (self.stride * 20) == 0:
            self.align_tries += 1
            A, n = align.estimate(small)
            if n >= align.MIN_INLIERS:
                self.A = A
        A = self.A if self.A is not None else np.eye(2, 3)
        obs = []
        for d in self.tracker.update(small, self.idx, t):
            foot = align.apply_points(A, np.array([[(d.x1 + d.x2) / 2, d.y2]]))[0]
            obs.append((d.track_id, d.type == "person", foot, d.y2 - d.y1))
        return self.core.update(t, obs)


def curve_from_tracks(tracks: list[Track], hz: float = RISK_HZ) -> list[tuple[float, float]]:
    """Risk over time from tracks already in the reference view, fed to RiskCore in time order.

    Used to calibrate RiskCore without a GPU and by the CPU demo server (one detector pass instead
    of two). Each update only sees observations up to its own time; the track positions themselves
    are smoothed over +-0.3 s, so unlike RiskModel this is not strictly causal (the submission's
    Part B always uses RiskModel).
    """
    by_time: dict[float, list] = defaultdict(list)
    for tr in tracks:
        for t, foot, h in zip(tr.t, tr.foot, tr.box_h):
            by_time[round(float(t) * hz) / hz].append((tr.id, tr.type == "person", foot, float(h)))
    core = RiskCore()
    return [(t, core.update(t, obs)) for t, obs in sorted(by_time.items())]
