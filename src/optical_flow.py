"""Optical flow motion field estimation.

Implements both sparse feature-based and dense differential optical flow algorithms:
1. Sparse Lucas-Kanade Optical Flow (KLT) with Shi-Tomasi corner initialization.
2. Dense Gunnar Farneback Optical Flow with HSV color-wheel direction and magnitude mapping.
3. Frame-to-frame motion vector field calculation and quantitative mean magnitude estimation.
"""

from typing import Tuple, List, Optional, Dict, Any
import cv2
import numpy as np


class OpticalFlowAnalyzer:
    """Computes sparse and dense optical flow for motion field analysis across consecutive video frames."""

    def __init__(self, method: str = "sparse", max_corners: int = 150,
                 quality_level: float = 0.03, min_distance: float = 10.0,
                 block_size: int = 7) -> None:
        """Args:

        method: 'sparse' (Lucas-Kanade) or 'dense' (Farneback).
        max_corners: Maximum Shi-Tomasi feature points to track in sparse mode.
        quality_level: Minimal accepted quality of image corners.
        min_distance: Minimum possible Euclidean distance between corners.
        block_size: Neighborhood window size.
        """
        self.method = method.lower()
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size

        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        # Shi-Tomasi feature detection parameters
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size
        )

        # State memory
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
        self.motion_mask: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Resets tracking history."""
        self.prev_gray = None
        self.prev_points = None
        self.motion_mask = None

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Processes current frame against previous frame to compute optical flow.

        Returns:
            Tuple of (visualization_frame, average_flow_magnitude, metrics_dict).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            if self.method == "sparse":
                self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
                self.motion_mask = np.zeros_like(frame)
            return frame.copy(), 0.0, {"num_points": 0, "avg_magnitude": 0.0}

        if self.method == "dense":
            vis_frame, avg_mag, metrics = self._process_dense(gray, frame)
        else:
            vis_frame, avg_mag, metrics = self._process_sparse(gray, frame)

        self.prev_gray = gray.copy()
        return vis_frame, avg_mag, metrics

    def _process_sparse(self, current_gray: np.ndarray,
                        current_bgr: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Computes Sparse Lucas-Kanade Optical Flow.

        Algorithm Explanation:
        1. What it does: Tracks Shi-Tomasi feature points across frames by solving the differential
           optical flow equation under the brightness constancy assumption.
        2. Mathematical Basis (Bruce D. Lucas & Takeo Kanade, 1981):
           Optical Flow Constraint: I_x * u + I_y * v + I_t = 0.
           Over a 21x21 neighborhood W:
           A * [u, v]^T = -b, where A = [I_x, I_y], b = I_t.
           Least Squares Solution: [u, v]^T = (A^T * A)^(-1) * A^T * (-b).
           Solved hierarchically over Gaussian pyramids to capture large displacements.
        """
        vis_frame = current_bgr.copy()

        # Re-detect features if point pool depleted
        if self.prev_points is None or len(self.prev_points) < 20:
            self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            self.motion_mask = np.zeros_like(current_bgr)

        if self.prev_points is None or len(self.prev_points) == 0:
            return vis_frame, 0.0, {"num_points": 0, "avg_magnitude": 0.0}

        # Calculate optical flow
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, current_gray, self.prev_points, None, **self.lk_params
        )

        avg_magnitude = 0.0
        good_count = 0
        magnitudes = []

        if curr_points is not None and status is not None:
            # Select good points (status == 1)
            good_curr = curr_points[status == 1]
            good_prev = self.prev_points[status == 1]

            for new_pt, old_pt in zip(good_curr, good_prev):
                a, b = new_pt.ravel()
                c, d = old_pt.ravel()

                dx = a - c
                dy = b - d
                mag = float(np.sqrt(dx ** 2 + dy ** 2))
                magnitudes.append(mag)

                # Draw persistent trajectory lines on motion mask
                cv2.line(self.motion_mask, (int(a), int(b)), (int(c), int(d)), (0, 230, 255), 2)
                cv2.circle(vis_frame, (int(a), int(b)), 4, (0, 0, 255), -1)

            good_count = len(good_curr)
            if len(magnitudes) > 0:
                avg_magnitude = float(np.mean(magnitudes))

            # Retain good points for next iteration
            self.prev_points = good_curr.reshape(-1, 1, 2)
        else:
            self.prev_points = None

        # Composite motion trails onto output frame
        vis_frame = cv2.add(vis_frame, self.motion_mask)

        # Decay motion mask slightly to keep visualization fresh
        self.motion_mask = cv2.addWeighted(self.motion_mask, 0.90, np.zeros_like(self.motion_mask), 0, 0)

        metrics = {
            "num_points": good_count,
            "avg_magnitude": round(avg_magnitude, 3),
            "max_magnitude": round(float(np.max(magnitudes)) if magnitudes else 0.0, 3)
        }
        return vis_frame, avg_magnitude, metrics

    def _process_dense(self, current_gray: np.ndarray,
                       current_bgr: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Computes Gunnar Farneback Dense Optical Flow with HSV color encoding.

        Algorithm Explanation:
        1. What it does: Computes motion displacement vector (u, v) for every single pixel in the image.
        2. Mathematical Basis (Gunnar Farneback, 2003):
           Approximates image neighborhoods by quadratic polynomial surfaces:
           f(x) approx x^T * A * x + b^T * x + c.
           Estimates displacement field d by matching coefficients across frames.
        3. Visualization:
           - Hue (H): Vector direction angle theta = arctan2(v, u).
           - Saturation (S): Set to 255 for vivid color saturation.
           - Value (V): Normalized motion magnitude sqrt(u^2 + v^2).
        """
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, current_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Compute magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        avg_magnitude = float(np.mean(mag))

        # HSV representation
        hsv = np.zeros_like(current_bgr)
        hsv[..., 0] = ang / 2.0  # OpenCV Hue [0, 180]
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # Blend with current frame for context
        vis_frame = cv2.addWeighted(current_bgr, 0.6, bgr_flow, 0.4, 0)

        metrics = {
            "avg_magnitude": round(avg_magnitude, 3),
            "max_magnitude": round(float(np.max(mag)), 3)
        }
        return vis_frame, avg_magnitude, metrics
