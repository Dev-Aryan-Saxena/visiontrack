"""Module: segmentation.py
Computer Vision Syllabus Mapping:
- Module 3 & Module 4: Image Segmentation (Thresholding, Region Growing,
  Mean-Shift, Pattern Clustering - K-Means).

Description:
Implements classical image segmentation algorithms:
1. Threshold-based segmentation (Otsu's optimal bimodal thresholding and adaptive Gaussian).
2. K-Means color-clustering segmentation (unsupervised feature-space partitioning).
3. Seeded Region Growing (spatial pixel connectivity and homogeneity criteria).
4. Mean-Shift segmentation (non-parametric feature space mode seeking).
"""

from typing import Tuple, List, Optional
import cv2
import numpy as np


def segment_by_threshold(gray_image: np.ndarray, method: str = "otsu",
                         global_thresh: float = 127.0) -> Tuple[np.ndarray, float]:
    """Performs threshold-based binary image segmentation.

    Algorithm Explanation:
    1. What it does: Partitions pixels into foreground (e.g., road markings, vehicles) and background
       based on intensity decision boundaries.
    2. Why it is used: Computationally ultra-lightweight; separates high-contrast road features.
    3. Mathematical Basis (Nobuyuki Otsu, 1979):
       Finds optimal threshold t* that maximizes the between-class variance:
       sigma_B^2(t) = omega_0(t) * omega_1(t) * [mu_0(t) - mu_1(t)]^2,
       where omega_i are class probabilities and mu_i are class means.
    4. Input: Grayscale image (H x W, uint8), method ('otsu', 'adaptive', 'global'), global threshold.
    5. Output: Tuple of (binary mask [0 or 255], calculated threshold value).
    """
    if len(gray_image.shape) != 2:
        raise ValueError("Input must be a single-channel grayscale image.")

    if method == "otsu":
        t_val, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary, float(t_val)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        return binary, -1.0
    elif method == "global":
        t_val, binary = cv2.threshold(gray_image, global_thresh, 255, cv2.THRESH_BINARY)
        return binary, float(t_val)
    else:
        raise ValueError(f"Unknown threshold method '{method}'. Choose 'otsu', 'adaptive', or 'global'.")


def segment_by_kmeans(bgr_image: np.ndarray, k: int = 4,
                      color_space: str = "RGB",
                      max_iterations: int = 20,
                      epsilon: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Performs unsupervised color clustering segmentation using the K-Means algorithm.

    Algorithm Explanation:
    1. What it does: Groups pixels into K distinct visual clusters based on color-space proximity.
    2. Why it is used: In road scenes, naturally segments distinct semantic components:
       asphalt road, green vegetation/curbs, sky, and vehicle bodies.
    3. Mathematical Basis (Lloyd's Algorithm):
       Minimizes within-cluster sum of squares (inertia):
       J = sum_{j=1}^K sum_{i in C_j} || x_i - mu_j ||^2
       Iterates two steps until convergence:
       - Assignment step: assign each pixel to closest centroid mu_j.
       - Update step: recompute centroid mu_j as the mean of all assigned pixels.
    4. Input: BGR color image (H x W x 3, uint8), number of clusters K.
    5. Output: Tuple of (segmented BGR image quantized to K colors, cluster labels array).
    """
    if len(bgr_image.shape) != 3:
        raise ValueError("Input must be a 3-channel color image.")

    if color_space.upper() == "LAB":
        converted = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    else:
        converted = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    pixel_values = converted.reshape((-1, 3)).astype(np.float32)

    # Termination criteria: Stop if max_iterations reached or centroids move less than epsilon
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, epsilon)
    # Run K-means with 3 random initializations (KMEANS_PP_CENTERS)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    # Map each pixel to its cluster center color
    centers = np.uint8(centers)
    segmented_flat = centers[labels.flatten()]
    segmented = segmented_flat.reshape(converted.shape)

    if color_space.upper() == "LAB":
        segmented_bgr = cv2.cvtColor(segmented, cv2.COLOR_LAB2BGR)
    else:
        segmented_bgr = cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)

    return segmented_bgr, labels.reshape(bgr_image.shape[:2])


def segment_by_region_growing(gray_image: np.ndarray,
                              seeds: Optional[List[Tuple[int, int]]] = None,
                              intensity_threshold: float = 15.0) -> np.ndarray:
    """Performs Seeded Region Growing segmentation.

    Algorithm Explanation:
    1. What it does: Starts from initial seed locations and iteratively aggregates neighboring pixels
       that meet a photometric homogeneity criterion.
    2. Why it is used: Captures spatially contiguous regions (e.g. road surface) sharing homogeneous
       reflectance even if illumination gradually drifts across the scene.
    3. Mathematical Basis: A neighbor pixel p is added to region R if |I(p) - mean(R)| <= T.
       The region mean is updated online as pixels are recruited.
    4. Input: Grayscale image (H x W, uint8), seed list [(x, y)], intensity difference threshold T.
    5. Output: Binary segmented region mask (H x W, uint8).
    """
    if len(gray_image.shape) != 2:
        raise ValueError("Input must be a single-channel grayscale image.")

    h, w = gray_image.shape
    segmented_mask = np.zeros((h, w), dtype=np.uint8)
    visited = np.zeros((h, w), dtype=bool)

    # If no seeds provided, auto-seed near bottom-center (standard road region)
    if not seeds:
        seeds = [(w // 2, int(h * 0.8)), (w // 4, int(h * 0.85)), (3 * w // 4, int(h * 0.85))]

    for seed in seeds:
        sx, sy = seed
        if sx < 0 or sx >= w or sy < 0 or sy >= h:
            continue
        if visited[sy, sx]:
            continue

        queue = [(sx, sy)]
        visited[sy, sx] = True
        region_pixels = [int(gray_image[sy, sx])]

        while queue:
            cx, cy = queue.pop(0)
            segmented_mask[cy, cx] = 255
            current_mean = np.mean(region_pixels)

            # 4-connectivity neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    if abs(int(gray_image[ny, nx]) - current_mean) <= intensity_threshold:
                        visited[ny, nx] = True
                        queue.append((nx, ny))
                        region_pixels.append(int(gray_image[ny, nx]))

    return segmented_mask


def segment_by_mean_shift(bgr_image: np.ndarray, spatial_radius: int = 15,
                          color_radius: int = 25, max_pyramid_level: int = 2) -> np.ndarray:
    """Performs Mean Shift image segmentation.

    Algorithm Explanation:
    1. What it does: Performs non-parametric mode seeking over the joint spatial-color domain (x, y, R, G, B).
    2. Why it is used: Does not require pre-specifying the number of clusters K (unlike K-Means).
       Preserves sharp edges while smoothing internal textures.
    3. Mathematical Basis (Comaniciu & Meer, 2002):
       Iteratively computes mean shift vector m(x) = [sum K((x-x_i)/h) x_i] / [sum K((x-x_i)/h)] - x,
       migrating each pixel toward local modes of the kernel density estimate.
    4. Input: BGR color image, spatial_radius (hs), color_radius (hr).
    5. Output: Segmented/filtered BGR image.
    """
    if len(bgr_image.shape) != 3:
        raise ValueError("Input must be a 3-channel color image.")

    return cv2.pyrMeanShiftFiltering(bgr_image, sp=spatial_radius, sr=color_radius,
                                     maxLevel=max_pyramid_level)
