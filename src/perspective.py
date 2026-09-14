"""Module: perspective.py
Computer Vision Syllabus Mapping:
- Module 1: Transformations (Orthogonal, Euclidean, Affine, Projective).
- Module 2: Depth Estimation and Multi-Camera Views (Perspective, Homography,
  Direct Linear Transformation - DLT, Rectification).

Description:
Implements 2D projective transformation and Inverse Perspective Mapping (IPM) to rectify
trapezoidal perspective distortions of road camera scenes into top-down bird's-eye views (BEV).
"""

from typing import Tuple, List, Optional
import cv2
import numpy as np


class PerspectiveTransformer:
    """Computes projective homography matrices and warps road images to bird's-eye view."""

    def __init__(self, src_points: Optional[np.ndarray] = None,
                 dst_points: Optional[np.ndarray] = None,
                 output_size: Optional[Tuple[int, int]] = None) -> None:
        """Args:

        src_points: 4 source points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] in perspective road image.
        dst_points: 4 destination points in rectilinear bird's-eye view plane.
        output_size: (width, height) of rectified bird's-eye image.
        """
        self.src_points = src_points
        self.dst_points = dst_points
        self.output_size = output_size
        self.homography_matrix: Optional[np.ndarray] = None
        self.inv_homography_matrix: Optional[np.ndarray] = None

        if src_points is not None and dst_points is not None:
            self.compute_homography(src_points, dst_points)

    def compute_homography(self, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
        """Computes 3x3 Projective Transformation (Homography) Matrix H.

        Algorithm Explanation:
        1. What it does: Maps coordinates from the camera projective image plane (x, y, 1)
           to the planar ground plane (x', y', 1).
        2. Why it is used: In road scenes, parallel lane dividers converge to a vanishing point due to
           perspective foreshortening. IPM restores geometric parallelism, enabling direct metric
           distance and longitudinal speed estimation.
        3. Mathematical Basis:
           Projective Homography has 8 degrees of freedom:
           [ x' ]       [ h11 h12 h13 ] [ x ]
           [ y' ] = H * [ h21 h22 h23 ] [ y ]
           [ 1  ]       [ h31 h32 h33 ] [ 1 ]
           Solved via Direct Linear Transformation (DLT) using 4 point correspondences.
        """
        src = np.float32(src_points)
        dst = np.float32(dst_points)

        self.homography_matrix = cv2.getPerspectiveTransform(src, dst)
        self.inv_homography_matrix = np.linalg.inv(self.homography_matrix)
        self.src_points = src
        self.dst_points = dst
        return self.homography_matrix

    def warp_to_birds_eye(self, image: np.ndarray,
                          output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Warps the perspective road image into bird's-eye view using homography H."""
        h, w = image.shape[:2]
        out_w, out_h = output_size if output_size is not None else (self.output_size or (w, h))

        if self.homography_matrix is None:
            # Generate default road trapezoid ROI if none configured
            # Source: trapezoid covering bottom 60% of road
            src = np.float32([
                [w * 0.15, h * 0.95],  # bottom-left
                [w * 0.85, h * 0.95],  # bottom-right
                [w * 0.58, h * 0.55],  # top-right
                [w * 0.42, h * 0.55]   # top-left
            ])
            # Destination: unwarped rectangle
            dst = np.float32([
                [out_w * 0.20, out_h * 0.95],
                [out_w * 0.80, out_h * 0.95],
                [out_w * 0.80, out_h * 0.05],
                [out_w * 0.20, out_h * 0.05]
            ])
            self.compute_homography(src, dst)

        return cv2.warpPerspective(image, self.homography_matrix, (out_w, out_h),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def warp_from_birds_eye(self, bev_image: np.ndarray,
                            target_size: Tuple[int, int]) -> np.ndarray:
        """Inversely warps bird's-eye representation back to the original camera perspective."""
        if self.inv_homography_matrix is None:
            raise ValueError("Homography has not been initialized.")
        return cv2.warpPerspective(bev_image, self.inv_homography_matrix, target_size,
                                  flags=cv2.INTER_LINEAR)

    def draw_roi_on_image(self, image: np.ndarray) -> np.ndarray:
        """Draws the 4-point perspective trapezoid ROI polygon on the original image for visual verification."""
        vis = image.copy()
        if self.src_points is not None:
            pts = self.src_points.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            for i, pt in enumerate(self.src_points):
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, (0, 0, 255), -1)
                cv2.putText(vis, f"P{i+1}", (int(pt[0]) + 8, int(pt[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return vis
