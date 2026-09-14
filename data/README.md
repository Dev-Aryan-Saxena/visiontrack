# VisionTrack Data Directory

This directory contains test images, videos, and stereo pairs for evaluating the VisionTrack pipeline.

## 1. Built-in Synthetic Sample Generator
To ensure the repository is 100% self-contained and reproducible without requiring multi-gigabyte downloads or copyrighted footage, VisionTrack includes a built-in synthetic data generator:

```bash
# Generate sample test road scene image, video, and stereo pair in data/
python main.py --generate-samples
```

This generates:
- `data/sample_road.jpg` — Synthetic road scene with multiple vehicle silhouettes, lane markings, horizon, and texture.
- `data/sample_traffic.mp4` — Synthetic 150-frame traffic video demonstrating multiple vehicles moving at varying speeds, background structures, and lane dividers.
- `data/left_stereo.jpg` & `data/right_stereo.jpg` — Binocular stereo image pair with horizontal disparity representing foreground objects and distant background for depth estimation.

## 2. Using Custom Real-World Data
You can also supply your own road scene data:
- **Images**: Any standard format (`.jpg`, `.jpeg`, `.png`, `.bmp`).
- **Videos**: Road/traffic recordings (`.mp4`, `.avi`, `.mov`).
- **Stereo Pairs**: Rectified binocular images with horizontal baseline.

Example:
```bash
python main.py --mode video --input data/my_traffic_clip.mp4
```
