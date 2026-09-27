"""Turn raw per-object segments into the submission format.

Same-class segments that overlap or nearly touch are merged into one (the organizers annotate two
simultaneous events of one class as a single segment); blips are dropped; times are clipped.
"""
from __future__ import annotations

MERGE_GAP_SEC = 1.0
MIN_LEN_SEC = 0.5


def merge(segments: list[tuple], gap: float = MERGE_GAP_SEC) -> list[tuple[float, float]]:
    """Union of (start, end, ...) segments; anything after start and end is ignored."""
    out: list[list[float]] = []
    for s, e, *_ in sorted(segments):
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def to_events(by_class: dict[str, list[tuple]], duration: float, enabled: list[str]) -> list[list]:
    events = []
    for label in enabled:
        for s, e in merge(by_class.get(label, [])):
            s, e = max(0.0, s), min(duration, e)
            if e - s >= MIN_LEN_SEC:
                events.append([round(s, 2), round(e, 2), label])
    return sorted(events)
