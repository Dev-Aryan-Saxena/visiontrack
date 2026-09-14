"""Module: stereo.py
Computer Vision Syllabus Mapping:
- Module 2: Depth Estimation And Multi-Camera Views (Binocular Stereopsis,
  Epipolar Geometry, Stereo Matching, Disparity Computation).

Description:
Implements classical binocular stereoscopic depth estimation:
1. Block Matching (StereoBM) for fast intensity-correlation matching.
2. Semi-Global Block Matching (StereoSGBM) with 1D dynamic programming path constraints.
3. Disparity normalization, invalid pixel masking, and false-color depth map visualization.
"""

from typing import Tuple, Optional, Dict, Any
import cv2
import numpy as np


class StereoDepthEstimator:
    """Computes dense disparity maps from rectified stereo image pairs."""

    def __init__(self, method: str = "sgbm", num_disparities: int = 64,
                 block_size: int = 9) -> None:
        """Args:

        method: 'sgbm' (Semi-Global Block Matching) or 'bm' (Block Matching).
        num_disparities: Maximum disparity range (must be positive multiple of 16).
        block_size: Matched block size (must be an odd number >= 1).
        """
        self.method = method.lower()
        self.num_disparities = num_disparities if num_disparities % 16 == 0 else 64
        self.block_size = block_size if block_size % 2 == 1 else 9

        self._init_matcher()

    def _init_matcher(self) -> None:
        """Initializes the OpenCV stereo matcher."""
        if self.method == "bm":
            self.matcher = cv2.StereoBM_create(
                numDisparities=self.num_disparities,
                blockSize=self.block_size
            )
        else:  # SGBM
            # Heiko Hirschmuller, 2008 - Semi-Global Matching (SGM)
            # Penalties P1 and P2 for disparity smoothness
            channels = 1
            p1 = 8 * channels * (self.block_size ** 2)
            p2 = 32 * channels * (self.block_size ** 2)

            self.matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=self.num_disparities,
                blockSize=self.block_size,
                P1=p1,
                P2=p2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )

    def compute_disparity(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Computes disparity map and false-color relative depth visualization.

        Algorithm Explanation:
        1. What it does: Finds pixel correspondences along rectified horizontal epipolar scanlines.
        2. Why it is used: Solves the binocular stereopsis correspondence problem to infer 3D depth.
        3. Mathematical Basis:
           Disparity d = x_left - x_right.
           Triangulation: Depth Z = (f * B) / d, where f is focal length, B is baseline distance.
           Objects closer to the stereo camera have larger disparities; distant background has small disparities.
           Academic Disclaimer: Without calibrated camera intrinsics (f) and extrinsics baseline (B),
           the output represents qualitative inverse relative depth, not metric meters.
        """
        # Convert to grayscale if needed
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) if len(left_img.shape) == 3 else left_img
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY) if len(right_img.shape) == 3 else right_img

        # Ensure identical dimensions
        if left_gray.shape != right_gray.shape:
            right_gray = cv2.resize(right_gray, (left_gray.shape[1], left_gray.shape[0]))

        # Compute raw disparity (fixed-point format with 4 fractional bits = 16x)
        raw_disp = self.matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0

        # Mask invalid disparities (negative values indicate occluded/unmatched pixels)
        valid_mask = raw_disp > 0
        valid_disparities = raw_disp[valid_mask]

        if len(valid_disparities) > 0:
            min_disp = float(np.min(valid_disparities))
            max_disp = float(np.max(valid_disparities))
            mean_disp = float(np.mean(valid_disparities))
        else:
            min_disp, max_disp, mean_disp = 0.0, 1.0, 0.0

        # Normalize valid disparities to [0, 255] uint8
        norm_disp = np.zeros_like(raw_disp, dtype=np.uint8)
        if max_disp > min_disp:
            scaled = (raw_disp - min_disp) / (max_disp - min_disp) * 255.0
            norm_disp[valid_mask] = np.clip(scaled[valid_mask], 0, 255).astype(np.uint8)

        # Generate false-color depth representation (INFERNO colormap: warm colors = close, cool = far)
        depth_colormap = cv2.applyColorMap(norm_disp, cv2.COLORMAP_INFERNO)
        depth_colormap[~valid_mask] = [0, 0, 0]  # Black for invalid/occluded pixels

        metrics = {
            "method": self.method,
            "num_disparities": self.num_disparities,
            "block_size": self.block_size,
            "min_disparity": round(min_disp, 2),
            "max_disparity": round(max_disp, 2),
            "mean_disparity": round(mean_disp, 2),
            "valid_pixel_ratio": round(float(np.sum(valid_mask)) / raw_disp.size, 4),
            "academic_note": "Uncalibrated relative disparity. True metric depth requires intrinsic matrix K and baseline B."
        }

        return norm_disp, depth_colormap, metrics
