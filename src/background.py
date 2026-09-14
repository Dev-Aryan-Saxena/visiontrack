"""Dynamic background modeling and foreground extraction.

Implements background subtraction algorithms and morphological post-processing:
1. Gaussian Mixture-based Background/Foreground Segmentation (MOG2).
2. K-Nearest Neighbors (KNN) Background Subtraction.
3. Morphological filtering (Erosion, Dilation, Opening, Closing) for noise elimination.
4. Moving contour extraction and minimum bounding rectangle filtering.
"""

from typing import Tuple, List, Dict, Any, Optional
import cv2
import numpy as np


class BackgroundSubtractor:
    """Encapsulates background modeling, morphological cleaning, and moving region detection."""

    def __init__(self, method: str = "mog2", history: int = 500, var_threshold: float = 16.0,
                 detect_shadows: bool = True, min_contour_area: float = 400.0) -> None:
        """Initializes the background subtractor.

        Algorithm Explanation:
        1. What it does: Maintains an adaptive probabilistic model of the background scene across frames.
        2. Why it is used: In video analysis of stationary road cameras, moving vehicles and pedestrians
           must be cleanly separated from static road, buildings, and swaying trees.
        3. Mathematical Basis (Zivkovic, 2004 - MOG2):
           Models each pixel intensity as a mixture of K Gaussians:
           P(X_t) = sum_{i=1}^K omega_{i,t} * N(X_t; mu_{i,t}, Sigma_{i,t}).
           MOG2 dynamically selects the number of Gaussians K for every pixel, adapting to lighting changes.
           Shadow pixels (which reduce intensity without changing chromaticity) are detected and flagged (value 127).
        """
        self.method = method.lower()
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.min_contour_area = min_contour_area

        if self.method == "mog2":
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=var_threshold, detectShadows=detect_shadows
            )
        elif self.method == "knn":
            self.subtractor = cv2.createBackgroundSubtractorKNN(
                history=history, dist2Threshold=var_threshold * 25.0, detectShadows=detect_shadows
            )
        else:
            raise ValueError(f"Unknown background subtraction method: {method}. Choose 'mog2' or 'knn'.")

        # Structuring elements for morphological filtering
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def apply(self, frame: np.ndarray, learning_rate: float = -1) -> Tuple[np.ndarray, np.ndarray]:
        """Computes raw foreground mask and morphologically cleaned binary mask.

        Morphological Post-Processing Explanation:
        1. Raw mask contains false-positive speckle noise (camera sensor noise, leaves, asphalt glare)
           and shadows (value 127).
        2. Binary thresholding isolates true foreground (value 255) from shadows and background (0).
        3. Morphological Opening (Erosion followed by Dilation):
           A circ B = (A ominus B) oplus B. Removes isolated small noise specks smaller than the kernel.
        4. Morphological Closing (Dilation followed by Erosion):
           A bullet B = (A oplus B) ominus B. Fills interior holes within vehicle bodies caused by
           uniform windshield or vehicle roof reflectance.
        5. Final Dilation:
           Restores vehicle boundaries that were slightly shrunk during erosion.
        """
        raw_mask = self.subtractor.apply(frame, learningRate=learning_rate)

        # Discard shadows: only retain true moving objects (pixel value == 255)
        _, binary_mask = cv2.threshold(raw_mask, 250, 255, cv2.THRESH_BINARY)

        # Morphological pipeline
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self.kernel_open, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel_close, iterations=2)
        cleaned_mask = cv2.dilate(closed, self.kernel_dilate, iterations=1)

        return raw_mask, cleaned_mask

    def extract_moving_regions(self, cleaned_mask: np.ndarray) -> List[Dict[str, Any]]:
        """Extracts bounding boxes and centroids of valid moving objects.

        Returns:
            List of dicts: [{"bbox": (x, y, w, h), "center": (cx, cy), "area": float}]
        """
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            # Aspect ratio sanity check (ignore extremely thin horizontal/vertical streaks)
            aspect_ratio = float(w) / (h + 1e-6)
            if aspect_ratio < 0.15 or aspect_ratio > 6.0:
                continue

            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)

            regions.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "center": (cx, cy),
                "area": float(area)
            })

        return regions

    def get_background_image(self) -> Optional[np.ndarray]:
        """Retrieves the learned static background model image if supported."""
        if hasattr(self.subtractor, "getBackgroundImage"):
            return self.subtractor.getBackgroundImage()
        return None
