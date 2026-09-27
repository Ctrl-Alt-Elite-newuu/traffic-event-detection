# Traffic event detection and accident anticipation — Ctrl+Alt+Elite

WIUT Hackathon 2026, Computer Vision track. Given a video from one fixed road camera, report every
traffic event as `[start_sec, end_sec, label]` (Part A) and a causal per-frame accident-risk score
(Part B). Website with EDA, results and a live demo: https://ctrl-alt-elite.sukoon.uz

## Install and run

```bash
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

- Python 3.10+. `requirements.txt` pins everything; the PyPI `torch` wheel for Linux includes CUDA
  and uses the GPU automatically (CPU works, but is slower than the 3× real-time budget).
- **Weights are in the repository** (`weights/yolo11s.pt`, 19 MB). No download step, no internet
  needed at run time.
- `run_submission.py` and `evaluate.py` are the organizers' files, unchanged.

## Approach

```
video ─► decode every 3rd frame, downscale 4K → 1920×1080
      ─► camera alignment to a reference view (SIFT + similarity transform)
      ─► YOLO11s detection (1280 px) + ByteTrack tracking ─► trajectories in the reference view
      ─► traffic-light colour read from pixels (2 Hz) ─► signal phases
      ─► per-class rules on trajectories + scene layout + signal ─► merge / clean segments ─► events
```

| Module | What it does | Learned or rule-based |
|---|---|---|
| `src/video.py` | Strided 4K reader (skipped frames are only grabbed), downscale to the reference size | — |
| `src/align.py` | The camera is re-mounted between recordings (up to ~70 px and 1° in the samples). Each video is aligned to `src/assets/reference.jpg` with SIFT and a RANSAC similarity transform | Rule-based |
| `src/tracking.py` | YOLO11s (COCO classes: person, bicycle, car, motorcycle, bus, truck) + ByteTrack | **Learned** (pretrained, not fine-tuned) |
| `src/trajectories.py` | Ground-contact point per object, smoothed; speed and heading | Rule-based |
| `src/signals.py` | Lamp colour of the median traffic light from fixed regions (saturation-based, works in sun and at dusk); phases over time | Rule-based |
| `src/scene.json`, `src/scene.py` | Crossings, stop lines, light positions, islands — annotated by hand once on the reference view (no scene description was provided) | Hand-annotated |
| `src/assets/scene_priors.npz` | Carriageway mask and usual direction of travel per cell, built from the sample videos' tracks by `scripts/build_scene_priors.py` | Statistics of the scene |
| `src/rules.py` | One rule per class (below) | Rule-based |
| `src/segments.py` | Merge same-class segments < 1 s apart, drop segments < 0.5 s | Rule-based |
| `src/risk.py` | Part B: time-to-collision between tracked vehicles, 5 Hz | Rule-based |

### Event rules (`src/rules.py`)

| Class | Rule |
|---|---|
| `jaywalking` | A walking pedestrian inside the carriageway (≥ 30 px from its edge), not on a crossing or island, for ≥ 1.5 s; riders of two-wheelers are excluded |
| `red_light` | A vehicle crosses the near carriageway's stop line in the direction of travel, after driving with the flow for 2 s, while the light has been red ≥ 1 s. The far carriageway's signal is not visible (in the samples more vehicles cross it on the visible "red" than on green), so it is not judged |
| `stopped_vehicle` | A vehicle (not a bus) stationary on the carriageway ≥ 10 s, measured in box heights per second so distant cars are not "stopped"; parked cars that never move and queues that leave within 10 s of a light change are excluded |
| `wrong_way` | A vehicle moving ≥ 120° against the usual direction where that direction is consistent, for ≥ 1.5 s |
| `failure_to_yield` | A vehicle driving through a crossing while a walking pedestrian is on the same crossing within 260 px |

Classes we do not predict are left out on purpose: a predicted class that never occurs in the test
set scores 0 and lowers the macro F1.

### Part B (`src/risk.py`)

`RiskEstimator.step` only sees the frames passed to it. Every 6th frame it runs YOLO11s (960 px) and
ByteTrack, fits each vehicle's velocity over the last 1.2 s, and scores each pair of **moving**
vehicles on converging headings by how soon (≤ 1.5 s) and how fast they would touch at constant
velocity. The score decays over ~1 s. On the sample videos (18 min, no accidents) this gives about
0.1 false alarms per minute; the two main false-alarm sources (a car passing a queued car in the next
lane, pedestrians passing cars at crossings) are excluded.

### Runtime

One decode pass for Part A plus the harness's decode for Part B. On a laptop CPU (no GPU) a 20 s 4K
clip takes 8.8× real time, ~90 % of it in YOLO; on a T4 the detector cost drops by an order of
magnitude, for an estimated ~2× real time against the 3× budget.

## Reproducing everything

```bash
python scripts/cache_tracks.py samples/*.MP4          # detection + tracking, cached to cache/
python scripts/cache_signals.py samples/*.MP4         # alignment + traffic-light phases
python scripts/build_scene_priors.py samples/*.MP4    # carriageway mask + flow directions
TRAFFIC_CACHE_DIR=cache python run_submission.py --videos samples --out predictions_samples.json
python scripts/export_site_data.py samples/*.MP4 --out ../traffic-event-detection-web/public/data \
    --predictions predictions_samples.json            # data and figures for the website
```

`TRAFFIC_CACHE_DIR` only lets Part A reuse the cached tracks and signal phases (they come from the
same code path); the official run leaves it unset and computes everything from the video.
`predictions_samples.json` was produced this way on a CPU.

## Determinism

Seeds are fixed (`random`, `numpy`, `torch`, all 0) in `solution.py` and `src/tracking.py`. There
is no sampling anywhere; YOLO and ByteTrack are deterministic on a given machine. GPU and CPU
inference can differ by floating-point noise, which can flip a borderline detection.

## Data, models and licences

| Item | Use | Licence |
|---|---|---|
| Sample videos from the organizers | EDA, scene annotation, scene priors, rule tuning | Organizers' terms |
| [YOLO11s](https://docs.ultralytics.com/models/yolo11/) by Ultralytics, pretrained on COCO | Detection (no fine-tuning) | AGPL-3.0 |
| [COCO](https://cocodataset.org) (via the pretrained weights) | Pre-training of the detector | CC BY 4.0 |
| ByteTrack (Ultralytics implementation) | Tracking | MIT (original) / AGPL-3.0 (Ultralytics) |

No other datasets were used.

## Repository layout

```
solution.py            entry point for the harness (detect_events, RiskEstimator)
run_submission.py      organizers' harness (unchanged)
evaluate.py            organizers' metric (unchanged)
src/                   pipeline, rules, risk model, scene layout and assets
scripts/               caching, scene priors, website export, trajectory plots
weights/               YOLO11s weights
demo/                  live-demo API (FastAPI + Docker) used by the website
deploy/                Docker Compose + Caddy for the team server
predictions_samples.json   our output on the four sample videos
```

## Team

| Member | Role and contributions |
|---|---|
| Asila Muxitdinova | TODO |
| Shaxnozaxon Abdusalomova | TODO |
| TODO | TODO |
