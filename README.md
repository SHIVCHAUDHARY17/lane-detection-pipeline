# Lane Detection Pipeline

> Classical computer vision pipeline for real-time lane detection in traffic video.  
> Built from scratch using OpenCV and NumPy — no deep learning required.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green)
![CI](https://github.com/SHIVCHAUDHARY17/lane-detection-pipeline/actions/workflows/ci.yml/badge.svg)
![Tests](https://img.shields.io/badge/tests-8%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

</div>

---

## Demo

<div align="center">

![Lane detection demo](docs/demo.gif)

*Real-time lane detection with polynomial fitting and temporal smoothing —  
green polygon marks the drivable lane area, stable across dashed markings.*

</div>

---

## Pipeline Architecture

<div align="center">

![Pipeline diagram](pipeline_diagram.png)

*Eight-stage classical CV pipeline — each stage builds on the previous one,  
transforming raw video into a stable annotated lane output.*

</div>

---

## What This Project Does

Takes a dashcam or traffic video as input and outputs a fully annotated video with:

- **Lane color isolation** — separates white and yellow lane pixels from road surface
- **Edge and line detection** — finds lane boundaries using gradient and voting methods
- **Bird's eye view transform** — geometrically corrects for camera angle using homography
- **Polynomial lane fitting** — fits smooth curves through lane pixels, handles curves
- **Temporal smoothing** — stabilises detection frame-to-frame across dashed lane gaps
- **Stage outputs** — every intermediate stage saves its own video for full traceability

---

## Stage-by-Stage Visual Progression

Each stage is independently runnable via `--stage N` and saves its own output video.

---

### Stage 1 — Raw input

<div align="center">

![](docs/stage1.jpg)

*Baseline — unprocessed dashcam video. No filtering, no annotations.  
This is the starting point every subsequent stage processes.*

</div>

---

### Stage 2 — HLS color masking

<div align="center">

![](docs/stage2.jpg)

*Lane-colored pixels isolated using HLS color space filtering.  
White lanes detected via grayscale threshold, yellow via HLS hue range.  
Everything else is blacked out — only lane candidates remain visible.*

</div>

---

### Stage 3 — Canny edge detection + ROI

<div align="center">

![](docs/stage3.jpg)

*Gaussian blur followed by Canny gradient-based edge detection.  
A triangular region of interest masks out the sky, trees, and surroundings —  
only road edges within the drivable zone are kept (shown in green).*

</div>

---

### Stage 4 — Hough line transform

<div align="center">

![](docs/stage4.jpg)

*Voting-based line detection across all edge pixels.  
Each edge pixel votes for every line it could belong to — peaks in the  
accumulator become detected line segments (shown in red).*

</div>

---

### Stage 5 — Line filtering + lane overlay

<div align="center">

![](docs/stage5.jpg)

*Raw Hough segments filtered by slope range, split into left and right lanes,  
and averaged using length-weighted averaging into two clean lane lines.  
A semi-transparent green polygon fills the detected lane area.*

</div>

---

### Stage 6 — Perspective transform (Bird's eye view)

<div align="center">

![](docs/stage6.jpg)

*A homography matrix maps four road points from camera view to a top-down  
bird's eye view. Left: source trapezoid on road. Right: warped BEV where  
lane markings appear parallel — the same technique used in ADAS and LiDAR-camera fusion.*

</div>

---

### Stage 7 — Polynomial lane fitting

<div align="center">

![](docs/stage7.jpg)

*Second-degree polynomial x = ay² + by + c fitted to lane pixels in BEV space.  
Left: fitted curves overlaid on BEV. Right: polynomial lanes projected back  
to camera view using the inverse homography matrix.*

</div>

---

### Stage 8 — Full pipeline with temporal smoothing

<div align="center">

![](docs/stage8.jpg)

*Complete pipeline with exponential moving average smoothing of polynomial  
coefficients across frames. Eliminates flickering during dashed lane gaps  
by blending current detections with recent history — same principle as Kalman filtering.*

</div>

---

## Key Concepts Implemented

### HLS color filtering
RGB values shift with lighting changes — a white lane in shadow looks grey in RGB.
HLS separates hue from lightness, so yellow stays yellow and white stays bright
regardless of illumination. This makes color-based lane isolation significantly
more robust across different times of day and road conditions.

### Canny edge detection
Gaussian blur first removes pixel noise, then gradient calculation finds brightness
transitions, and double-threshold hysteresis keeps only strong connected edges.
This gives clean, thin lane boundaries rather than thick noisy blobs.

### Hough line transform
Instead of connecting every pair of edge pixels (computationally explosive),
Hough uses a voting accumulator in polar space (ρ, θ). Each pixel votes for
every line it could lie on — where votes concentrate, a real line exists.
The `threshold` parameter controls the minimum vote count to count as a line.

### Perspective transform (homography)
A 3×3 matrix computed from four point correspondences maps every pixel from
camera view to bird's eye view. Lanes that converge to a vanishing point
appear parallel in BEV — enabling geometric reasoning about curvature and
real-world distances. The inverse matrix projects results back to camera view.

### Polynomial lane fitting
Hough lines only detect straight segments and fail on curves. A second-degree
polynomial fitted to lane pixel coordinates in BEV handles both straight and
curved roads. Fitting x as a function of y (not y of x) avoids instability
for near-vertical lanes.

### Temporal smoothing
Dashed lane markings cause pixel dropout between dashes — the polynomial fit
returns `None` when too few pixels exist. Exponential moving average blends
current coefficients with a recent history buffer. When detection fails,
memory holds the last valid fit for up to `max_age` frames before discarding.

---

## Project Structure

```
lane-detection-pipeline/
│
├── src/
│   ├── video_io.py          # Video capture and writer utilities
│   ├── color_mask.py        # HLS and grayscale color filtering
│   ├── edge_detection.py    # Canny edge detection and ROI masking
│   ├── hough.py             # Hough line transform and drawing
│   ├── lane_lines.py        # Line filtering, averaging, lane overlay
│   ├── perspective.py       # Perspective transform and BEV warp
│   ├── lane_fit.py          # Polynomial fitting and projection
│   └── smoother.py          # Temporal smoothing across frames
│
├── config/
│   └── default.yaml         # All tunable parameters — no hardcoding
│
├── tests/
│   ├── test_color_mask.py
│   ├── test_edge_detection.py
│   └── test_smoother.py
│
├── docs/                    # Screenshots and demo GIF
├── data/                    # Input video files
├── output/                  # Generated output videos
├── main.py                  # CLI entry point
├── conftest.py              # pytest path configuration
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## Setup

```bash
git clone https://github.com/SHIVCHAUDHARY17/lane-detection-pipeline.git
cd lane-detection-pipeline

python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
source venv/bin/activate          # Linux / Mac

pip install -r requirements.txt
```

---

## Usage

```bash
# Run a specific pipeline stage
python main.py --stage 8

# Run all stages sequentially
for i in 1 2 3 4 5 6 7 8; do python main.py --stage $i; done

# Run tests
pytest tests/ -v
```

Each stage saves its output automatically:

| Stage | Output file |
|-------|-------------|
| 1 | `output/day1_raw_passthrough.mp4` |
| 2 | `output/day2_color_mask.mp4` |
| 3 | `output/day3_edges_roi.mp4` |
| 4 | `output/day4_hough_lines.mp4` |
| 5 | `output/day5_lane_overlay.mp4` |
| 6 | `output/day6_bev_transform.mp4` |
| 7 | `output/day7_poly_fit.mp4` |
| 8 | `output/day8_full_pipeline.mp4` |

---

## Configuration

All parameters live in `config/default.yaml` — nothing is hardcoded in the source.

```yaml
color_mask:
  white_threshold: 200           # Grayscale brightness cutoff for white lanes
  hls_yellow_lower: [15,100,100] # HLS lower bound for yellow detection
  hls_yellow_upper: [35,255,255] # HLS upper bound for yellow detection

canny:
  blur_kernel: 5                 # Gaussian blur kernel size (must be odd)
  low_threshold: 50              # Canny lower threshold
  high_threshold: 150            # Canny upper threshold

hough:
  rho: 1
  theta_degrees: 1
  threshold: 40                  # Minimum votes to count as a line
  min_line_length: 100
  max_line_gap: 50

smoother:
  alpha: 0.7                     # Blend factor (higher = trust current frame more)
  max_age: 10                    # Frames to hold memory after detection dropout
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| OpenCV 4.13 | Image processing and video I/O |
| NumPy | Array operations and polynomial fitting |
| PyYAML | Config file parsing |
| pytest | Unit testing |
| GitHub Actions | CI — runs tests on every push |

---

## Author

<div align="center">

**Shiv Jayant Chaudhary**  
Computer Vision and Machine Learning Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-shiv1716-blue?logo=linkedin)](https://linkedin.com/in/shiv1716)
[![GitHub](https://img.shields.io/badge/GitHub-SHIVCHAUDHARY17-black?logo=github)](https://github.com/SHIVCHAUDHARY17)

</div>


