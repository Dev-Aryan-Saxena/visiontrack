"""Module: preprocessing.py
Computer Vision Syllabus Mapping:
- Module 1: Digital Image Formation and Low Level Processing (Fundamentals of Image Formation,
  Geometric Transformations, Convolution and Filtering, Noise Reduction and Restoration).

Description:
Provides foundational low-level image processing operations including color space conversions,
linear and non-linear spatial convolution filters, geometric transformations, and noise injection
for restoration benchmarking.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts a BGR image to grayscale using luminance weighting.

    Algorithm Explanation:
    1. What it does: Converts a 3-channel color image into a single-channel intensity image.
    2. Why it is used: Reduces data dimensionality from 3 channels to 1 while preserving perceptual
       luminance, which is essential for low-level edge detection, frequency analysis, and feature extraction.
    3. Mathematical Basis: ITU-R BT.601 standard: Y = 0.299 * R + 0.587 * G + 0.114 * B, reflecting
       human spectral sensitivity peaks in the green wavelength band.
    4. Input: BGR color image (H x W x 3, uint8).
    5. Output: Grayscale intensity image (H x W, uint8).
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is None or empty.")
    if len(image.shape) == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_filter(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Applies a 2D Gaussian smoothing filter.

    Algorithm Explanation:
    1. What it does: Convolves the image with an isotropic 2D Gaussian kernel.
    2. Why it is used: Suppresses high-frequency additive Gaussian noise before gradient or edge computation,
       preventing spurious edge artifacts.
    3. Mathematical Basis: Kernel G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2)).
       The 2D Gaussian is separable into two 1D Gaussian convolutions, reducing complexity from O(K^2) to O(2K).
    4. Input: Image (H x W or H x W x C, uint8), kernel_size (must be odd > 0), sigma (standard deviation).
    5. Output: Smoothed image (same shape and dtype).
    """
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError(f"Kernel size must be a positive odd integer, got {kernel_size}.")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


def apply_median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Applies a non-linear median filter for impulse noise removal.

    Algorithm Explanation:
    1. What it does: Replaces each pixel value with the median intensity of its local neighborhood.
    2. Why it is used: Eliminates salt-and-pepper (impulse) noise without blurring sharp step edges,
       unlike linear low-pass filters that smear step transitions.
    3. Mathematical Basis: For neighborhood W around (x, y), I_out(x, y) = median({I(p) for p in W}).
    4. Input: Image (H x W or H x W x C, uint8), kernel_size (positive odd integer).
    5. Output: Denoised image (same shape and dtype).
    """
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError(f"Kernel size must be a positive odd integer, got {kernel_size}.")
    return cv2.medianBlur(image, kernel_size)


def apply_bilateral_filter(image: np.ndarray, diameter: int = 9, sigma_color: float = 75.0,
                           sigma_space: float = 75.0) -> np.ndarray:
    """Applies an edge-preserving bilateral filter.

    Algorithm Explanation:
    1. What it does: Smooths flat image textures while strictly preserving sharp object boundaries.
    2. Why it is used: In road scenes, road surface texture or asphalt grain should be smoothed
       without destroying vehicle outlines or lane boundaries.
    3. Mathematical Basis: Combines geometric spatial closeness G_sigma_s(||p-q||) and photometric range
       similarity G_sigma_r(||I(p)-I(q)||) in the convolution kernel weights.
    4. Input: Image (H x W x C or H x W, uint8), diameter, sigma_color, sigma_space.
    5. Output: Edge-preserved smoothed image.
    """
    return cv2.bilateralFilter(image, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def apply_box_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Applies a standard uniform averaging (box) filter.

    Algorithm Explanation:
    1. What it does: Convolves the image with a normalized constant kernel where all weights are 1 / (K^2).
    2. Why it is used: Demonstrates classical unweighted spatial mean convolution.
    3. Input: Image (uint8), kernel_size (positive odd integer).
    4. Output: Mean-filtered image.
    """
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError(f"Kernel size must be a positive odd integer, got {kernel_size}.")
    return cv2.blur(image, (kernel_size, kernel_size))


def resize_image(image: np.ndarray, target_width: Optional[int] = None,
                 target_height: Optional[int] = None, preserve_aspect: bool = True) -> np.ndarray:
    """Resizes an image with optional aspect ratio preservation.

    Algorithm Explanation:
    1. What it does: Applies 2D geometric scaling using area or bilinear interpolation.
    2. Why it is used: Normalizes resolution for predictable CPU computation speeds in real-time pipelines.
    3. Input: Image (H x W x C or H x W), target dimensions.
    4. Output: Rescaled image.
    """
    h, w = image.shape[:2]
    if target_width is None and target_height is None:
        return image.copy()

    if preserve_aspect:
        if target_width is not None and target_height is not None:
            scale = min(target_width / w, target_height / h)
        elif target_width is not None:
            scale = target_width / w
        else:
            scale = target_height / h
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
    else:
        new_w = target_width if target_width is not None else w
        new_h = target_height if target_height is not None else h

    interpolation = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def apply_affine_transform(image: np.ndarray, angle_deg: float = 0.0, scale: float = 1.0,
                           translation: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Applies an affine transformation (Euclidean/Similarity: rotation, scaling, and translation).

    Algorithm Explanation:
    1. What it does: Transforms coordinates via x' = A x + t, preserving parallelism of straight lines.
    2. Why it is used: Demonstrates Module 1 affine and Euclidean geometric transformation fundamentals.
    3. Input: Image, rotation angle in degrees, scaling factor, translation vector (tx, ty).
    4. Output: Affine transformed image.
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    # Get 2x3 rotation and scaling matrix
    m = cv2.getRotationMatrix2D(center, angle_deg, scale)
    # Add translation component
    m[0, 2] += translation[0]
    m[1, 2] += translation[1]
    return cv2.warpAffine(image, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, std: float = 20.0) -> np.ndarray:
    """Adds synthetic Gaussian noise to test restoration filters."""
    noisy = image.astype(np.float32) + np.random.normal(mean, std, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(image: np.ndarray, salt_prob: float = 0.02, pepper_prob: float = 0.02) -> np.ndarray:
    """Adds synthetic impulse (salt and pepper) noise to test median filtering."""
    noisy = image.copy()
    rnd = np.random.rand(*image.shape[:2])
    # Salt (white pixels)
    noisy[rnd < salt_prob] = 255
    # Pepper (black pixels)
    noisy[(rnd >= salt_prob) & (rnd < (salt_prob + pepper_prob))] = 0
    return noisy
