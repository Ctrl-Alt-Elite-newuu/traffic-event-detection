"""Event rules: trajectories + scene layout + signal phases -> time segments per class.

Every rule returns (start_sec, end_sec, track_id) per offending object; segments.py merges them per
class and cleans them, and the track id lets the website draw the object that caused the event.
All coordinates are reference-view pixels (1920x1080), speeds are px/s.
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from . import config
from .scene import Scene, side_of_line
from .signals import SignalTimeline
from .trajectories import Track, runs

Segment = tuple[float, float, int]

# --- jaywalking ---
JAY_CROSSWALK_MARGIN = 35.0    # px: crowds spill a little past the stripes; that is still the crossing
JAY_EDGE_PX = 15.0             # feet this close to the bottom of the frame are cut off: position unknown
IN_VEHICLE = 0.6               # a "person" within this many vehicle heights of a vehicle is inside it
JAY_MIN_SEC = 1.5
JAY_MIN_TRAVEL = 25.0          # px: a "person" that never moves is a pole or a false detection
RIDER_RADIUS = 45.0            # px: a person this close to a two-wheeler is its rider
RIDER_SPEED = 220.0            # px/s: faster than anyone walks

# --- red-light running ---
RED_GRACE_SEC = 1.0            # red must have been on this long (the tail of amber is not a violation)
RED_END_GRACE_SEC = 1.0        # ...and not about to turn green (the light is read at 2 Hz)
RED_EXIT_PX = 260.0            # the event ends when the vehicle is this far past the stop line...
RED_MAX_SEC = 8.0              # ...or after this long, or when its track ends
RED_APPROACH_SEC = 2.0         # the vehicle must have been driving with the flow this long before the line
RED_APPROACH_COS = 0.8

# --- stopped vehicle ---
STOP_SPEED = 0.06              # box heights per second: distant cars move few pixels, so pixels lie
STOP_MIN_SEC = 10.0
PARKED_TRAVEL = 1.0            # box heights: a track that never travels further than this is parked
QUEUE_RELEASE_SEC = 10.0       # a stop that ends this soon after any light change was a queue at a signal
QUEUE_CREEP_SEC = 3.0          # ...or this soon before one (queues creep forward before green)
QUEUE_RADIUS = 2.5             # box heights: another stopped vehicle this close means a queue...
QUEUE_SHARE = 0.5              # ...for at least this share of the stop (works for signals we cannot see)

# --- wrong-way driving ---
WRONG_MIN_SPEED = 60.0
WRONG_MIN_CONSISTENCY = 0.8    # only where the flow prior is one-directional
WRONG_MIN_SAMPLES = 20
WRONG_COS = -0.5               # heading at least 120 degrees off the usual direction
WRONG_MIN_SEC = 1.5

# --- failure to yield ---
YIELD_MIN_SPEED = 30.0         # a vehicle waiting on the crossing is not driving through it
YIELD_PED_RADIUS = 260.0       # px: pedestrian close enough along the crossing to be in conflict
YIELD_MARGIN = 0.0
YIELD_PED_SPEED = 15.0
YIELD_LENGTH = 0.6             # box heights: the ground point is inside the crossing for less time than
YIELD_MAX_PAD_SEC = 1.0        # the whole vehicle (front enters -> rear leaves); pad by that length


def _in_polygon(poly: np.ndarray, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
    return np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= -margin for x, y in pts])


def _positions_by_time(tracks: list[Track], keep) -> dict[float, list[tuple[Track, np.ndarray, float]]]:
    """time -> [(track, foot, speed)] for tracks passing `keep`; detections share frame timestamps."""
    index: dict[float, list] = defaultdict(list)
    for tr in tracks:
        if keep(tr):
            for t, p, s in zip(tr.t, tr.foot, tr.speed):
                index[round(float(t), 2)].append((tr, p, s))
    return index


def jaywalking(tracks: list[Track], scene: Scene) -> list[Segment]:
    riders = _positions_by_time(tracks, lambda tr: tr.type in ("motorcycle", "bicycle"))
    vehicles: dict[float, list[tuple[np.ndarray, float]]] = defaultdict(list)   # drivers seen through glass
    for v in tracks:
        if v.is_vehicle and v.type not in ("motorcycle", "bicycle"):
            for t, p, h in zip(v.t, v.foot, v.box_h):
                vehicles[round(float(t), 2)].append((p, float(h)))
    out = []
    for tr in tracks:
        if tr.type != "person" or np.ptp(tr.foot, axis=0).max() < JAY_MIN_TRAVEL:
            continue
        on_crossing = np.zeros(len(tr.t), bool)
        for poly in scene.crosswalks.values():
            on_crossing |= _in_polygon(poly, tr.foot, JAY_CROSSWALK_MARGIN)
        on_road = np.array([scene.on_road(x, y, inner=True) and not scene.in_parking(x, y) for x, y in tr.foot])
        riding = tr.speed > RIDER_SPEED
        inside_vehicle = np.zeros(len(tr.t), bool)
        for i, (t, p) in enumerate(zip(tr.t, tr.foot)):
            key = round(float(t), 2)
            if any(np.linalg.norm(p - q) < RIDER_RADIUS for _, q, _ in riders.get(key, [])):
                riding[i] = True
            inside_vehicle[i] = any(np.linalg.norm(p - q) < IN_VEHICLE * h for q, h in vehicles.get(key, []))
        if riding.mean() > 0.3:          # this "pedestrian" is a rider for most of the track
            continue
        at_edge = tr.foot[:, 1] > config.REF_H - JAY_EDGE_PX
        mask = on_road & ~on_crossing & ~riding & ~inside_vehicle & ~at_edge
        out += [(float(tr.t[i]), float(tr.t[j]), tr.id) for i, j in runs(mask, tr.t, JAY_MIN_SEC, max_gap=0.5)]
    return out


def red_light(tracks: list[Track], scene: Scene, signal: SignalTimeline) -> list[Segment]:
    out = []
    for tr in tracks:
        if not tr.is_vehicle or tr.type == "bicycle":
            continue
        for name, line in scene.stop_lines.items():
            if scene.stop_line_signal.get(name) is None:
                continue                        # the signal for this line is not visible from the camera
            (x0, y0), (x1, y1) = line
            seg = np.array([x1 - x0, y1 - y0])
            length = float(np.linalg.norm(seg))
            normal = np.array([-seg[1], seg[0]]) / length
            s = np.array([side_of_line(line, x, y) for x, y in tr.foot]) / length
            along = ((tr.foot - line[0]) @ seg) / length ** 2
            for i in range(1, len(tr.t)):
                if np.sign(s[i - 1]) == np.sign(s[i]) or not -0.05 <= along[i] <= 1.05:
                    continue
                # must cross in the usual direction of travel there (not a vehicle turning across the line)
                flow, consistency, _ = scene.flow_at(*tr.foot[i])
                if consistency < 0.6 or np.dot(tr.heading[i], flow) < 0.5:
                    continue
                t_cross = float(tr.t[i])
                phase = signal.phase_at(t_cross)
                if signal.red_for(t_cross) < RED_GRACE_SEC or phase["end"] - t_cross < RED_END_GRACE_SEC:
                    continue
                before = (tr.t >= t_cross - RED_APPROACH_SEC) & (tr.t < t_cross)
                if tr.t[0] > t_cross - RED_APPROACH_SEC or (tr.heading[before] @ flow).mean() < RED_APPROACH_COS:
                    continue                    # turned in from the side street on its own green
                past = np.abs((tr.foot[i:] - tr.foot[i]) @ normal)
                beyond = np.flatnonzero(past > RED_EXIT_PX)
                t_end = float(tr.t[i + beyond[0]]) if len(beyond) else tr.end
                out.append((t_cross, min(t_end, t_cross + RED_MAX_SEC), tr.id))
    return out


def stopped_vehicle(tracks: list[Track], scene: Scene, signal: SignalTimeline) -> list[Segment]:
    stopped_at: dict[float, list[tuple[Track, np.ndarray]]] = defaultdict(list)   # time -> stopped vehicles
    for o in tracks:
        if o.is_vehicle:
            for t, p, still in zip(o.t, o.foot, o.speed < STOP_SPEED * np.median(o.box_h)):
                if still:
                    stopped_at[round(float(t), 2)].append((o, p))
    out = []
    for tr in tracks:
        if not tr.is_vehicle or tr.type == "bus":      # buses dwell at the stop by design
            continue
        size = np.median(tr.box_h)
        if np.linalg.norm(tr.foot - np.median(tr.foot, axis=0), axis=1).max() < PARKED_TRAVEL * size:
            continue                                    # never really moves: parked, not "stopped"
        still = tr.speed < STOP_SPEED * size
        on_road = np.array([scene.on_road(x, y) and not scene.in_parking(x, y) for x, y in tr.foot])
        changes = np.array(signal.changes())
        for i, j in runs(still & on_road, tr.t, STOP_MIN_SEC, max_gap=2.0):
            t0, t1 = float(tr.t[i]), float(tr.t[j])
            near_change = (changes <= t1 + QUEUE_CREEP_SEC) & (changes >= t1 - QUEUE_RELEASE_SEC)
            if t1 < tr.end - 0.5 and len(changes) and near_change.any():
                continue                                # drove off around a light change: a queue
            queued = [any(o is not tr and np.linalg.norm(q - p) < QUEUE_RADIUS * size
                          for o, q in stopped_at.get(round(float(t), 2), []))
                      for t, p in zip(tr.t[i:j + 1], tr.foot[i:j + 1])]
            if np.mean(queued) >= QUEUE_SHARE:
                continue                                # stopped among other stopped vehicles: a queue
            if t1 >= tr.end - 0.5 and signal.state_at(t1) != "green":
                continue                                # still waiting when the track ends, at a red light
            out.append((t0, t1, tr.id))
    return out


def wrong_way(tracks: list[Track], scene: Scene) -> list[Segment]:
    out = []
    for tr in tracks:
        if not tr.is_vehicle:
            continue
        against = np.zeros(len(tr.t), bool)
        for k, (p, h, s) in enumerate(zip(tr.foot, tr.heading, tr.speed)):
            if s < WRONG_MIN_SPEED:
                continue
            flow, consistency, n = scene.flow_at(*p)
            against[k] = consistency >= WRONG_MIN_CONSISTENCY and n >= WRONG_MIN_SAMPLES and np.dot(h, flow) < WRONG_COS
        out += [(float(tr.t[i]), float(tr.t[j]), tr.id) for i, j in runs(against, tr.t, WRONG_MIN_SEC, max_gap=0.5)]
    return out


def failure_to_yield(tracks: list[Track], scene: Scene) -> list[Segment]:
    walkers = _positions_by_time(tracks, lambda tr: tr.type == "person")
    walkers = {t: [w for w in ws if w[2] > YIELD_PED_SPEED] for t, ws in walkers.items()}
    out = []
    for tr in tracks:
        if not tr.is_vehicle or tr.type == "bicycle":
            continue
        for poly in scene.crosswalks.values():
            inside = _in_polygon(poly, tr.foot)
            for i, j in runs(inside, tr.t, 0.2, max_gap=0.3):
                if tr.speed[i:j + 1].mean() < YIELD_MIN_SPEED:
                    continue
                conflict = False
                for k in range(i, j + 1):
                    for _, q, _ in walkers.get(round(float(tr.t[k]), 2), []):
                        if (cv2.pointPolygonTest(poly, (float(q[0]), float(q[1])), True) >= -YIELD_MARGIN
                                and np.linalg.norm(q - tr.foot[k]) < YIELD_PED_RADIUS):
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    pad = min(YIELD_MAX_PAD_SEC, YIELD_LENGTH * float(np.median(tr.box_h[i:j + 1]))
                              / max(float(tr.speed[i:j + 1].mean()), 1e-6))
                    out.append((float(tr.t[i]) - pad, float(tr.t[j]) + pad, tr.id))
    return out


def detect_all(tracks: list[Track], scene: Scene, signal: SignalTimeline) -> dict[str, list[Segment]]:
    return {
        "jaywalking": jaywalking(tracks, scene),
        "red_light": red_light(tracks, scene, signal),
        "stopped_vehicle": stopped_vehicle(tracks, scene, signal),
        "wrong_way": wrong_way(tracks, scene),
        "failure_to_yield": failure_to_yield(tracks, scene),
    }
