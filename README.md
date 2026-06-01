# Real-Time Exercise Repetition Counter & Form Analyser

A computer vision system for automatic exercise analysis using real-time pose estimation. The system counts repetitions, detects postural errors, and provides instant corrective feedback for three exercises: push-ups, jumping jacks, and lunges.

Built as a final project for the **Computer Vision** course at Faculdade de Ciências, Universidade do Porto (2025/2026).

---

## Features

### Push-Up Analysis
- Repetition counting via elbow angle FSM (DOWN: θ < 90°, UP: θ > 155°)
- Multi-parameter quality validation per repetition:
  - **Depth** — minimum elbow angle ≤ 96° at DOWN phase
  - **Lockout** — full extension with angle ≥ 130° at UP phase
  - **Hip sag** — shoulder-hip-ankle alignment detection with two severity levels (warning and critical)
- Adaptive thresholds calibrated over first 60 frames

### Jumping Jack Analysis
- Dual-criteria synchronisation detection:
  - **Arm elevation** — shoulder-wrist vertical angle > 130°
  - **Foot separation** — normalised ankle distance > 0.20
- OPEN → CLOSED transition requires both criteria simultaneously, validating bilateral motor coordination
- False positive rate < 5% with adaptive threshold calibration

### Lunge Analysis
- Bilateral working leg identification in real time via hierarchical multi-criteria system:
  1. Angular difference (weight 10): |θL − θR| > 30° → lower angle = working leg
  2. Shin verticality (weight 5): minimum knee-ankle horizontal offset
  3. Knee height (weight 2): lower knee position
  4. 3D depth (weight 1): Z-coordinate tiebreaker
- Temporal confirmation over 2 consecutive frames to prevent oscillation
- Works from any camera angle (front, side, back)
- Detects insufficient depth (θknee > 110°) and knee-over-toe fault

---

## Architecture

Each module (`push_up.py`, `jumping_jack.py`, `lunge.py`) follows a unified pipeline:

```
Video capture (webcam or file)
       ↓
MediaPipe Pose (33 3D landmarks)
       ↓
Joint angle & Euclidean distance computation
       ↓
Binary FSM (UP ↔ DOWN)
       ↓
Repetition quality classification
       ↓
Real-time visual feedback overlay
```

---

## Getting started

### Prerequisites

```bash
pip install -r src/requirements.txt
```

### Run a module

```bash
# Push-up analyser (webcam)
python src/push_up.py

# Push-up analyser (video file)
python src/push_up.py --source src/pushup1.mp4

# Jumping jack analyser
python src/jumping_jack.py

# Lunge analyser
python src/lunge.py
```

### Controls

| Key | Action |
|---|---|
| P | Pause |
| R | Reset counter |
| C | Recalibrate thresholds |
| Q | Quit |

---

## Tech stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=flat&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)

---

## Results

Tested on 720p-1080p video at 25-30 FPS:

| Exercise | Key metric |
|---|---|
| Push-up | Depth and lockout validated; hip sag detected with two severity levels |
| Jumping Jack | Bilateral synchronisation validated; false positive rate < 5% |
| Lunge | Bilateral working leg identification validated on complete and incomplete movements |

**Limitations:** occlusions above 30% degrade precision (mitigated by visibility > 0.5 threshold); ballistic movements above 60°/frame smoothed with a 5-frame temporal buffer.