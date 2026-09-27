"""Render an annotated, web-sized copy of each video: every tracked road user boxed, the ones in an
event highlighted with the class name, the active events and the traffic-light state in a banner.

    python scripts/render_annotated.py samples/*.MP4 --out ../traffic-event-detection-web/public/data/results
Writes <out>/<video>.mp4 (960x540, H.264, 10 fps). Needs ffmpeg and the cached tracks.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.pipeline import analyse  # noqa: E402
from src.rules import detect_all  # noqa: E402
from src.scene import Scene  # noqa: E402
from src.tracking import load_tracks  # noqa: E402
from src.video import iter_frames  # noqa: E402

OUT_W, OUT_H = 960, 540
# BGR, matching the website's class families
FAMILY_BGR = {
    "jaywalking": (122, 175, 27), "red_light": (52, 104, 235), "failure_to_yield": (52, 104, 235),
    "stopped_vehicle": (214, 120, 42), "wrong_way": (167, 58, 74),
}
LIGHT_BGR = {"red": (59, 59, 208), "yellow": (25, 178, 250), "green": (12, 163, 12), "unknown": (130, 130, 130)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scene = Scene.load()

    for video in args.videos:
        name = Path(video).name
        a = analyse(video, cache_dir=config.CACHE_DIR)
        events = detect_all(a.tracks, scene, a.signal)
        # (track id) -> [(start, end, label)] so boxes can be coloured while their event is running
        by_track: dict[int, list] = defaultdict(list)
        for label, segs in events.items():
            for s, e, tid in segs:
                by_track[tid].append((s, e, label))
        dets_by_frame = defaultdict(list)
        for d in load_tracks(config.CACHE_DIR / f"{name}.tracks.csv"):
            dets_by_frame[d.frame].append(d)

        out = args.out / f"{Path(video).stem}.mp4"
        fps_out = a.info.fps / config.DETECT_STRIDE
        ff = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}",
             "-r", f"{fps_out:.4f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
        sx = OUT_W / config.REF_W
        for idx, t, frame in iter_frames(video, config.DETECT_STRIDE):
            img = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
            active = []
            for d in dets_by_frame.get(idx, []):
                hits = [lab for s, e, lab in by_track.get(d.track_id, []) if s <= t <= e]
                p0, p1 = (int(d.x1 * sx), int(d.y1 * sx)), (int(d.x2 * sx), int(d.y2 * sx))
                if hits:
                    color = FAMILY_BGR[hits[0]]
                    cv2.rectangle(img, p0, p1, color, 2)
                    cv2.putText(img, hits[0].replace("_", " "), (p0[0], max(12, p0[1] - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                    active += hits
                else:
                    cv2.rectangle(img, p0, p1, (200, 200, 200), 1)
            cv2.rectangle(img, (0, 0), (OUT_W, 26), (0, 0, 0), -1)
            state = a.signal.state_at(t)
            cv2.circle(img, (14, 13), 7, LIGHT_BGR[state], -1)
            text = f"{int(t // 60)}:{t % 60:04.1f}  light {state}"
            if active:
                text += "  |  " + ", ".join(sorted(set(lab.replace("_", " ") for lab in active)))
            cv2.putText(img, text, (28, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            ff.stdin.write(img.tobytes())
        ff.stdin.close()
        ff.wait()
        print(f"[{name}] -> {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
