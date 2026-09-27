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

## Engineering and security practices

How the code gets from a laptop to the judges and to the public demo.

**Repository and CI**
- Two branches: all work lands on `dev`; `main` changes only through pull requests. A ruleset on
  `main` blocks force pushes and deletion and requires a pull request with passing checks.
- Every push and pull request runs CI (`.github/workflows/ci.yml`): a clean install from
  `requirements.txt`, an import check of the harness interface, and `evaluate.py --validate-only` on
  the committed predictions. The demo's Docker image is built on every push, so a broken build is
  caught on `dev`, but it is only published and deployed from `main`.
- Workflows run with least-privilege tokens (`contents: read`; `packages: write` only for the job
  that publishes the image). Dependencies are pinned to exact versions.
- No secrets or large data in git: the 19 GB of sample videos and all caches are ignored; the deploy
  key lives only in GitHub's encrypted Actions secrets.

**Server (`deploy/`)**
- The demo shares a server with other projects, so it is isolated rather than given the machine:
  a dedicated `traffic` user (no password, SSH key only, used by nothing else) owns the two
  directories it deploys to, with its own ed25519 key used only by GitHub Actions.
- The API container listens on `127.0.0.1:8001` only; the one public entry point is the existing
  Caddy, which terminates HTTPS (Let's Encrypt certificate, automatic renewal, HTTP → HTTPS
  redirect). Adding our site was one site block; the server's other sites were not touched.
- Every deploy pins the image to the exact commit (`DEMO_TAG=<git sha>`), so what runs is always
  traceable to a commit and can be rolled back by re-running an older workflow.

**Demo API hardening (`demo/app.py`)**
- Uploads are limited twice: 320 MB at the proxy and 300 MB in the app, which streams the upload
  to disk in 1 MB chunks and aborts past the limit instead of buffering it in memory.
- Only `.mp4` files are accepted, and the video is opened and its duration checked (≤ 2 min)
  before any work is queued.
- One job runs at a time, so a burst of uploads queues instead of exhausting a 2-core server.
- Uploaded videos are deleted as soon as processing ends, including on errors; nothing is kept.
- CORS is restricted to the site's own origin, and a failing job returns an error message
  instead of taking the server down.

**Known gaps** (what we would do next): the container still runs as root inside Docker; GitHub
Actions are pinned to major-version tags rather than commit SHAs; there is no per-client rate
limit beyond the single-job queue. Membership of the `docker` group is effectively root on the
host, so the dedicated user separates our deploys from the other projects but is not a security
boundary on its own.

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
