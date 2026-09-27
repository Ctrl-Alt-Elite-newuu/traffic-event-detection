"""Traffic-light state from fixed lamp regions of the frame."""
from __future__ import annotations

import cv2
import numpy as np

LAMP_HUE = {"red": ((0, 12), (165, 180)), "yellow": ((12, 35),), "green": ((40, 95),)}


def lamp_score(frame: np.ndarray, box: tuple[int, int, int, int], lamp: str) -> float:
    """Fraction of strongly coloured pixels of the lamp's own hue inside its box.

    Saturation, not brightness, separates a lit lamp: in direct sun the grey housing is
    brighter than the LED, but it is colourless.
    """
    x0, y0, x1, y1 = box
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_ok = np.zeros(h.shape, bool)
    for lo, hi in LAMP_HUE[lamp]:
        hue_ok |= (h >= lo) & (h < hi)
    lit = hue_ok & (s > 120) & (v > 60)
    return float(lit.mean())


def light_state(frame: np.ndarray, lamps: dict[str, tuple[int, int, int, int]], min_score: float = 0.03) -> str:
    """'red' / 'yellow' / 'green' for the brightest lamp, or 'unknown' if none is lit."""
    scores = {lamp: lamp_score(frame, box, lamp) for lamp, box in lamps.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= min_score else "unknown"


class SignalTimeline:
    """Traffic-light phases over a video: [{"start", "end", "state"}], contiguous and covering it."""

    MIN_PHASE_SEC = 1.5  # shorter runs (flashing green, glare, occlusion) are absorbed by the neighbour

    def __init__(self, phases: list[dict]):
        self.phases = phases

    @classmethod
    def from_samples(cls, samples: list[tuple[float, str]], duration: float) -> "SignalTimeline":
        raw: list[dict] = []
        for t, state in samples:
            if raw and raw[-1]["state"] == state:
                raw[-1]["end"] = t
            else:
                raw.append({"start": t, "end": t, "state": state})
        merged: list[dict] = []
        for p in raw:
            if merged and (p["end"] - p["start"] < cls.MIN_PHASE_SEC or p["state"] == merged[-1]["state"]):
                merged[-1]["end"] = p["end"]
            else:
                merged.append(dict(p))
        for a, b in zip(merged, merged[1:]):
            a["end"] = b["start"]
        if merged:
            merged[0]["start"] = 0.0
            merged[-1]["end"] = duration
        return cls([{"start": round(p["start"], 2), "end": round(p["end"], 2), "state": p["state"]} for p in merged])

    def state_at(self, t: float) -> str:
        for p in self.phases:
            if p["start"] <= t < p["end"]:
                return p["state"]
        return "unknown"

    def phase_at(self, t: float) -> dict | None:
        for p in self.phases:
            if p["start"] <= t < p["end"]:
                return p
        return None

    def red_for(self, t: float) -> float:
        """How long the light has been red at time t (0 if it is not red)."""
        p = self.phase_at(t)
        return t - p["start"] if p and p["state"] == "red" else 0.0

    def next_green_after(self, t: float) -> float | None:
        for p in self.phases:
            if p["state"] == "green" and p["start"] >= t:
                return p["start"]
        return None

    def changes(self) -> list[float]:
        """Times at which the light changes state."""
        return [p["start"] for p in self.phases[1:]]
