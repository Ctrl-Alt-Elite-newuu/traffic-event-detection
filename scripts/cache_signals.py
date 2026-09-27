"""Read the traffic-light state over whole videos and cache the phases to cache/<video>.signals.json.

    python scripts/cache_signals.py samples/*.MP4

Each phase is {"start": s, "end": s, "state": "red" | "yellow" | "green" | "unknown"}; flashing green
shows up as short green/unknown alternations and is merged into green.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import align, config  # noqa: E402
from src.scene import Scene  # noqa: E402
from src.signals import SignalTimeline, light_state  # noqa: E402
from src.video import iter_frames, probe  # noqa: E402

SAMPLE_HZ = 2.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    args = ap.parse_args()
    for video in args.videos:
        info = probe(video)
        A, inliers = align.estimate_for_video(video, info.fps)
        lamps = Scene.load().to_video(A).lights["main"]
        stride = max(1, round(info.fps / SAMPLE_HZ))
        samples = [(t, light_state(frame, lamps)) for _, t, frame in iter_frames(video, stride)]
        out = config.CACHE_DIR / f"{Path(video).name}.signals.json"
        out.write_text(json.dumps({"video": Path(video).name, "align": A.round(5).tolist(), "align_inliers": inliers,
                                   "phases": SignalTimeline.from_samples(samples, info.duration).phases}, indent=1))
        print(f"[{Path(video).name}] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
