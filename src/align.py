"""Align each video to the reference view.

The camera is fixed, but between recordings it is re-mounted slightly
differently (up to ~70 px shift and ~1 deg rotation in the samples). All scene
geometry lives in the reference view (assets/reference.jpg), so for every video
we estimate a similarity transform video -> reference from SIFT matches on
static structure and map detections through it.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .video import iter_frames

REFERENCE_IMAGE = Path(__file__).with_name("assets") / "reference.jpg"
N_PROBE_FRAMES = 5       # frames spread over the first ~20 s; the best-matching one wins
MIN_INLIERS = 30         # below this the estimate is not trusted and identity is used

_sift = cv2.SIFT_create(4000)
_matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
_ref_features = None


def _gray(img: np.ndarray) -> np.ndarray:
    """Contrast-normalised grey image, so daylight and dusk frames match."""
    return cv2.createCLAHE(3.0, (8, 8)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


def _reference_features():
    global _ref_features
    if _ref_features is None:
        _ref_features = _sift.detectAndCompute(_gray(cv2.imread(str(REFERENCE_IMAGE))), None)
    return _ref_features


def estimate(frame: np.ndarray) -> tuple[np.ndarray, int]:
    """2x3 similarity transform mapping `frame` (reference resolution) onto the reference view."""
    kr, dr = _reference_features()
    k, d = _sift.detectAndCompute(_gray(frame), None)
    if d is None or len(k) < 10:
        return np.eye(2, 3), 0
    good = [p[0] for p in _matcher.knnMatch(d, dr, k=2) if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < MIN_INLIERS:
        return np.eye(2, 3), 0
    src = np.float32([k[m.queryIdx].pt for m in good])
    dst = np.float32([kr[m.trainIdx].pt for m in good])
    A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if A is None:
        return np.eye(2, 3), 0
    return A, int(inl.sum())


def estimate_for_video(path: str, fps: float) -> tuple[np.ndarray, int]:
    """Best transform over a few frames spread across the first seconds of the video."""
    stride = max(1, round(fps * 4))
    best = (np.eye(2, 3), 0)
    for i, (_, _, frame) in enumerate(iter_frames(path, stride)):
        if i >= N_PROBE_FRAMES:
            break
        A, n = estimate(frame)
        if n > best[1]:
            best = (A, n)
    return best if best[1] >= MIN_INLIERS else (np.eye(2, 3), best[1])


def apply_points(A: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map Nx2 video points into the reference view."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return pts @ A[:, :2].T + A[:, 2]


def invert(A: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(A)
