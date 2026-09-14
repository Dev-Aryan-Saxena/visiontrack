# VisionTrack

### Road Scene Analysis and Motion Tracking

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Passing](https://img.shields.io/badge/tests-15%20passed-brightgreen.svg)](tests/)
[![Report](https://img.shields.io/badge/report-technical%20specification-blue.svg)](report/project_report.md)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-red.svg)](https://opencv.org/)

VisionTrack is a standalone command-line Computer Vision system that analyzes road scenes from images and videos. It implements a multi-stage visual analysis pipeline combining classical image processing, projective geometry, dynamic background modeling, multi-object tracking, and differential optical flow estimation.

---

## Architecture & Pipelines

### Video Pipeline (Primary)

```
Input Video
    │
    ▼
Frame Preprocessing (Resizing, Grayscale, Denoising)
    │
    ▼
Motion / Background Detection (MOG2 + Morphological Cleaning)
    │
    ▼
Object Detection (YOLOv8 / HOG-SVM / Motion-Blob)
    │
    ▼
Object Tracking (Centroid + IoU Matching with Persistent IDs)
    │
    ▼
Optical Flow (Lucas-Kanade Sparse / Farneback Dense)
    │
    ▼
Motion Analysis (dx, dy, Displacement, px/frame Speed)
    │
    ▼
Outputs: Annotated Video + CSV Trajectories + JSON Summary
```

### Image Pipeline

```
Input Image
    │
    ▼
Preprocessing & Denoising (Gaussian, Median, Bilateral)
    │
    ▼
Enhancement (CLAHE, Histogram Equalization)
    │
    ▼
Edge Detection (Canny, LoG)
    │
    ▼
Feature Extraction (Hough Lines, Harris Corners, HOG)
    │
    ▼
Segmentation (K-Means, Otsu, Region Growing, Mean-Shift)
    │
    ▼
Outputs: Filtered, Edges, Features, Segmentation, Summary
```

---

## Core Features

- **Headless CLI Execution**: Operates completely from the terminal without graphical display dependencies (`cv2.imshow` is not required).
- **Video Motion Analysis**:
  - Background modeling using Gaussian Mixture Models (MOG2).
  - Morphological noise cleaning (opening and closing).
  - Multi-target tracking with persistent unique IDs (`Car #1`, `Truck #2`, `Person #3`).
  - Kinematic metrics: $\Delta x$, $\Delta y$, instantaneous speed, and cumulative displacement in pixels/frame.
  - Optical flow motion field estimation (Lucas-Kanade sparse feature tracking or Farneback dense flow).
- **Image Scene Analysis**:
  - Denoising (Gaussian, median, bilateral filtering).
  - Contrast enhancement via CLAHE and histogram equalization.
  - Canny edge detection.
  - Geometric feature localization (Hough transform lane lines and Harris corner points).
  - Multi-method segmentation (K-Means color clustering, Otsu thresholding, Region Growing).
- **Perspective Normalization**:
  - Direct Linear Transformation (DLT) homography mapping road planes to bird's-eye view (BEV).
- **Stereo Vision**:
  - Binocular disparity map and relative depth visualization using Semi-Global Block Matching (StereoSGBM).
- **Self-Contained Testing**:
  - Built-in synthetic test media generator produces calibrated test images, traffic video, and stereo pairs with one command.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Dev-Aryan-Saxena/visiontrack.git
cd visiontrack

# 2. Create and activate virtual environment
python3 -m venv .venv

# Linux / macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **CPU-Only Environments**: If running on a CPU-only machine without an NVIDIA GPU, you can avoid downloading multi-gigabyte CUDA packages by installing the CPU build of PyTorch before installing requirements:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

---

## Model Setup & Detection Engines

VisionTrack supports multiple object detection engines:

1. **Lightweight YOLOv8 (Default when installed)**:
   - Uses `ultralytics` YOLOv8n (nano, ~6 MB).
   - Automatically initializes and downloads the lightweight model if requested.
2. **Classical Motion-Blob Detection (Zero-Setup Fallback)**:
   - Uses MOG2 background subtraction paired with morphological contour analysis.
   - Runs 100% offline out-of-the-box with zero external downloads required.
3. **HOG + Linear SVM**:
   - Built-in detector for locating pedestrians in road scenes.

If YOLO weights are missing or offline, VisionTrack automatically logs an informative message and falls back to classical motion detection.

---

## Quickstart & Usage

### 1. Generate Sample Test Data
Generate sample media in `data/` to test a fresh setup:
```bash
python main.py --generate-samples
```
*Creates `data/sample_image.jpg`, `data/sample_traffic.mp4`, `data/left.jpg`, and `data/right.jpg`.*

### 2. Run Full End-to-End Demo
Executes both image analysis and video tracking in one command:
```bash
python main.py --mode demo
```

### 3. Video Analysis & Tracking
```bash
python main.py --mode video --input data/sample_traffic.mp4
```

Output in terminal:
```
Processing video...
Frame:   15/120 | Tracked:  4 | Motion:  6.16 px/frame | Optical Flow:  2.45 px/frame
Frame:   30/120 | Tracked:  5 | Motion:  3.41 px/frame | Optical Flow:  2.62 px/frame
Frame:   45/120 | Tracked:  3 | Motion:  3.45 px/frame | Optical Flow:  3.28 px/frame
...
[VisionTrack] Total Unique Objects Tracked: 11
[VisionTrack] Tracking Data: outputs/video/2026-09-14_142200/tracking.csv
[VisionTrack] Summary Report: outputs/video/2026-09-14_142200/summary.json
```

**Video Options**:
```bash
# Use Dense Farneback optical flow
python main.py --mode video --input data/sample_traffic.mp4 --optical-flow dense

# Speed up processing on long clips using frame skipping and resizing
python main.py --mode video --input data/sample_traffic.mp4 --frame-skip 2 --resize 480
```

### 4. Image Analysis
```bash
python main.py --mode image --input data/sample_image.jpg
```

With bird's-eye view perspective transformation:
```bash
python main.py --mode image --input data/sample_image.jpg --perspective
```

Choose segmentation method:
```bash
python main.py --mode image --input data/sample_image.jpg --segment otsu
```

### 5. Stereo Depth Estimation
```bash
python main.py --mode stereo --left data/left.jpg --right data/right.jpg
```

### 6. Interactive CLI Help
```bash
python main.py --help
```

---

## Output Structure

All outputs are saved to clean, timestamped directories:

### Video Mode
```
outputs/
└── video/
    └── 2026-09-14_142200/
        ├── annotated_video.mp4  # Bounding boxes, IDs, trajectories, HUD
        ├── optical_flow.mp4     # Motion vector field visualization
        ├── tracking.csv         # Frame-by-frame trajectory coordinates & speeds
        └── summary.json         # Aggregate tracking metrics & video parameters
```

**CSV Schema (`tracking.csv`)**:
```csv
frame_idx,object_id,class_name,center_x,center_y,box_x1,box_y1,box_x2,box_y2,displacement_dx,displacement_dy,instantaneous_speed_px_per_frame,cumulative_distance_px
6,1,vehicle,160,295,138,284,183,306,0.0,0.0,0.0,0.0
7,1,vehicle,160,292,126,280,194,305,0.0,-3.0,3.0,3.0
8,1,vehicle,161,290,128,278,194,303,1.0,-2.0,2.24,5.24
```

### Image Mode
```
outputs/
└── image/
    └── 2026-09-14_142148/
        ├── original.jpg      # Input image
        ├── filtered.jpg      # Preprocessed & contrast-enhanced image
        ├── edges.jpg         # Canny edge map
        ├── features.jpg      # Hough lines (green) & Harris corners (red) overlay
        ├── segmentation.jpg  # Segmented image (e.g. K-Means color clusters)
        ├── birds_eye.jpg     # Rectified road view (if --perspective enabled)
        └── summary.json      # Resolution, feature counts, distribution metrics
```

### Stereo Mode
```
outputs/
└── stereo/
    └── 2026-09-14_142225/
        ├── disparity_map.png # Normalized grayscale disparity map
        ├── depth_map.png     # Inferno false-color relative depth visualization
        └── summary.json      # Disparity statistics & valid pixel ratio
```

---

## Technical Approach

### 1. Preprocessing & Enhancement
- **Gaussian Smoothing**: Convolves with a 2D isotropic Gaussian kernel to attenuate high-frequency sensor noise.
- **Median Filtering**: Non-linear rank-order filter that eliminates impulse noise while preserving sharp step edges.
- **Contrast Limited Adaptive Histogram Equalization (CLAHE)**: Divides the image into contextual tiles, clips local histograms to avoid noise amplification, and interpolates across tile boundaries.

### 2. Feature Extraction
- **Canny Edge Detection**: Four-stage detector using Gaussian smoothing, Sobel gradient computation, Non-Maximum Suppression (NMS) along the gradient direction, and hysteresis thresholding.
- **Hough Line Transform**: Maps edge points into accumulator space ($\rho = x\cos\theta + y\sin\theta$) to locate linear lane boundaries.
- **Harris Corner Detection**: Computes the auto-correlation structure tensor $M = \sum w(x,y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$ and thresholded response $R = \det(M) - k(\text{trace}(M))^2$.
- **Histogram of Oriented Gradients (HOG)**: Evaluates 9-bin gradient orientation histograms over local spatial cells with block normalization.

### 3. Image Segmentation
- **Otsu's Thresholding**: Computes an optimal global intensity threshold by maximizing the between-class variance $\sigma_B^2(t)$.
- **K-Means Clustering**: Unsupervised color-space partitioning that minimizes within-cluster Euclidean sum of squares.
- **Seeded Region Growing**: Recursively groups neighboring pixels that satisfy a similarity threshold around seed locations.

### 4. Background Modeling & Morphology
- **MOG2 (Mixture of Gaussians)**: Models each pixel as an adaptive mixture of Gaussians, dynamically updating weights to separate static road background from moving vehicles.
- **Morphological Cleaning**: Applies morphological opening ($\text{erosion} \to \text{dilation}$) to remove isolated noise pixels, followed by closing ($\text{dilation} \to \text{erosion}$) to fill holes in vehicle masks.

### 5. Multi-Object Tracking & Motion Estimation
- Associates detections with active tracks using a cost matrix based on Euclidean centroid distance and bounding box Intersection-over-Union (IoU).
- Maintains persistent IDs across consecutive frames and handles temporary occlusions with a configurable grace period.
- Computes frame-to-frame displacement $(\Delta x, \Delta y)$, instantaneous velocity $\sqrt{\Delta x^2 + \Delta y^2}$ in pixels/frame, and cumulative distance.

### 6. Optical Flow
- **Lucas-Kanade (Sparse)**: Tracks Shi-Tomasi feature points across frames by solving the differential optical flow constraint equation via least squares over local neighborhoods.
- **Farneback (Dense)**: Computes two-frame dense displacement fields using polynomial expansion, rendered using HSV directional color mapping.

### 7. Perspective Transformation (BEV)
- Solves for the $3 \times 3$ homography matrix $H$ via Direct Linear Transformation (DLT) from 4 planar point correspondences, unwarping the road surface into a rectilinear top-down view.

### 8. Stereo Depth Estimation
- Matches rectified horizontal epipolar scanlines using Semi-Global Block Matching (StereoSGBM) with smoothness energy minimization.

---

## Limitations & Considerations

1. **Pixel-Based Motion Units**: Displacements and speeds are reported strictly in **pixels/frame** units. Converting to physical units (km/h) requires extrinsic camera calibration (mounting height, pitch, focal length, and known physical road markers).
2. **Stationary Camera Assumption**: Background subtraction assumes a fixed camera view. Ego-motion or panning cameras require global motion compensation.
3. **Uncalibrated Stereo Depth**: Stereo disparity maps reflect relative, inverse-depth relationships rather than absolute metric distances without known camera baseline and focal length.

---

## Testing

Run the automated test suite with `pytest`:

```bash
pytest -v
```

Tests cover:
- Grayscale conversion, Gaussian/median filtering, and FFT frequency-domain reconstruction.
- Otsu thresholding, K-Means clustering, and region growing.
- Centroid tracking logic, IoU calculation, and object lifecycle states.
- Sparse Lucas-Kanade and dense Farneback optical flow computation.

---

## Repository Structure

```
visiontrack/
├── main.py               # CLI entrypoint
├── requirements.txt      # Python dependencies
├── README.md             # Documentation
├── LICENSE               # MIT License
├── .gitignore            # Git ignore rules
│
├── src/                  # Core modules
│   ├── preprocessing.py  # Denoising and filtering
│   ├── enhancement.py    # Histogram equalization & CLAHE
│   ├── frequency.py      # 2D FFT & frequency filtering
│   ├── features.py       # Canny, Hough lines, Harris, HOG, SIFT
│   ├── segmentation.py   # Otsu, K-Means, Region Growing
│   ├── background.py     # MOG2 background subtraction & morphology
│   ├── detection.py      # Detection engines (YOLOv8, HOG-SVM, motion)
│   ├── tracking.py       # Multi-object tracker & kinematics
│   ├── optical_flow.py   # Lucas-Kanade & Farneback optical flow
│   ├── perspective.py    # Homography & bird's-eye view
│   ├── stereo.py         # Stereo disparity & depth estimation
│   └── utils.py          # Video I/O, synthetic data generator, serializers
│
├── data/                 # Sample media & test datasets
│   └── README.md
│
├── outputs/              # Timestamped run directories
│   └── .gitkeep
│
├── report/               # Technical system report & specifications
│   └── project_report.md
│
└── tests/                # Automated pytest suite
    ├── test_preprocessing.py
    ├── test_segmentation.py
    ├── test_tracking.py
    └── test_optical_flow.py
```

---

## License

This project is licensed under the [MIT License](LICENSE).
