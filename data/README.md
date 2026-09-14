# VisionTrack Data Directory

This directory stores test images, videos, and stereo pairs for VisionTrack.

## Generating Sample Media
VisionTrack includes a built-in generator to produce test media for quick evaluation:

```bash
python main.py --generate-samples
```

This generates:
- `data/sample_image.jpg` — Synthetic road scene with lane markings, horizon, and vehicles.
- `data/sample_traffic.mp4` — 120-frame traffic video with multiple moving vehicles.
- `data/left.jpg` & `data/right.jpg` — Stereo pair with calibrated horizontal disparity.

## Using Real Traffic Data
You can place your own road scene data directly in this folder:
- **Images**: `.jpg`, `.jpeg`, `.png`, `.bmp`
- **Videos**: `.mp4`, `.avi`, `.mov`
- **Stereo Pairs**: Rectified left and right images
