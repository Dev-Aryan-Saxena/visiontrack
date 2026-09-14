"""Image enhancement and histogram processing.

Implements intensity transformations, global and local adaptive histogram equalization (CLAHE),
power-law (gamma) corrections, linear contrast stretching, and histogram distribution analysis.
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Strictly headless backend
import matplotlib.pyplot as plt


def equalize_histogram_gray(gray_image: np.ndarray) -> np.ndarray:
    """Applies global histogram equalization on a grayscale image.

    Algorithm Explanation:
    1. What it does: Flattens the image histogram so that intensity levels are uniformly distributed.
    2. Why it is used: Enhances global contrast, especially in road scenes captured under poor,
       flat, or hazy lighting conditions.
    3. Mathematical Basis: Uses the normalized Cumulative Distribution Function (CDF) as a transfer function:
       T(k) = round((L - 1) * sum_{j=0}^k (n_j / N)), mapping intensities k in [0, L-1].
    4. Input: Grayscale image (H x W, uint8).
    5. Output: Histogram-equalized image (H x W, uint8).
    """
    if len(gray_image.shape) != 2:
        raise ValueError("Input must be a single-channel grayscale image.")
    return cv2.equalizeHist(gray_image)


def equalize_histogram_color(bgr_image: np.ndarray) -> np.ndarray:
    """Applies histogram equalization on a color image using the YCrCb color space.

    Algorithm Explanation:
    1. What it does: Converts BGR to YCrCb, equalizes ONLY the luminance channel (Y),
       and preserves chromaticity channels (Cr, Cb) before converting back.
    2. Why it is used: Equalizing RGB channels independently corrupts color balance (color bleeding);
       operating on luminance separates lightness from chromatic information.
    3. Input: BGR color image (H x W x 3, uint8).
    4. Output: Contrast-enhanced color image (H x W x 3, uint8).
    """
    if len(bgr_image.shape) != 3 or bgr_image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel BGR color image.")
    ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def apply_clahe(gray_image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Algorithm Explanation:
    1. What it does: Computes histogram equalization over localized contextual grid tiles and limits
       amplification by clipping histogram bins above clip_limit before redistributing them.
    2. Why it is used: Standard global histogram equalization tends to overamplify background noise in
       near-homogeneous regions (e.g. road tarmac or sky). CLAHE prevents noise blowout while boosting
       local details (e.g. lane markers and shadowed vehicle undercarriages).
    3. Mathematical Basis: Tiles image into M x N contextual regions, clips histograms, computes local CDFs,
       and interpolates boundaries using bilinear interpolation to eliminate tile artifacts.
    4. Input: Grayscale image (H x W, uint8), clip_limit, tile_grid_size.
    5. Output: CLAHE-enhanced image (H x W, uint8).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(gray_image.shape) == 2:
        return clahe.apply(gray_image)
    elif len(gray_image.shape) == 3:
        # Apply to luminance channel in LAB space
        lab = cv2.cvtColor(gray_image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        raise ValueError("Unsupported image shape.")


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Applies power-law (gamma) transformation: s = c * r^gamma.

    Algorithm Explanation:
    1. What it does: Non-linearly stretches dark or bright intensity levels.
    2. Why it is used: Compares dynamic range. Gamma < 1.0 brightens underexposed night road scenes;
       Gamma > 1.0 darkens overexposed bright sunlight scenes.
    3. Mathematical Basis: s = 255 * (r / 255)^gamma, implemented via precomputed 256-element lookup table (LUT).
    4. Input: Image (uint8), gamma (> 0).
    5. Output: Gamma-corrected image (uint8).
    """
    if gamma <= 0:
        raise ValueError("Gamma value must be strictly positive.")
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(image, lut)


def apply_contrast_stretching(gray_image: np.ndarray, lower_percentile: float = 2.0,
                              upper_percentile: float = 98.0) -> np.ndarray:
    """Applies linear percentile contrast stretching (min-max normalization).

    Algorithm Explanation:
    1. What it does: Linearly stretches the active intensity range to span the full [0, 255] spectrum.
    2. Why it is used: Recovers dynamic range lost during camera sensor quantization without distorting tones.
    3. Mathematical Basis: s = clip((r - a) / (b - a) * 255, 0, 255), where a and b are the low and high percentiles.
    4. Input: Grayscale image (H x W, uint8), lower_percentile, upper_percentile.
    5. Output: Contrast-stretched image (uint8).
    """
    if len(gray_image.shape) != 2:
        raise ValueError("Input must be a single-channel grayscale image.")
    a = float(np.percentile(gray_image, lower_percentile))
    b = float(np.percentile(gray_image, upper_percentile))
    if b <= a:
        return gray_image.copy()
    stretched = (gray_image.astype(np.float32) - a) / (b - a) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def compute_histogram_metrics(gray_image: np.ndarray) -> Dict[str, float]:
    """Computes statistical metrics of the image intensity distribution.

    Metrics include:
    - Mean intensity: central tendency of brightness.
    - Standard deviation: overall image contrast indicator.
    - Entropy: Shannon information measure of randomness/detail (bits/pixel).
    """
    if len(gray_image.shape) != 2:
        raise ValueError("Input must be a single-channel grayscale image.")

    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten()
    pdf = hist / (hist.sum() + 1e-12)

    mean_val = float(np.mean(gray_image))
    std_val = float(np.std(gray_image))

    # Shannon Entropy: -sum(p * log2(p))
    non_zero = pdf[pdf > 0]
    entropy_val = float(-np.sum(non_zero * np.log2(non_zero)))

    return {
        "mean_intensity": round(mean_val, 2),
        "std_contrast": round(std_val, 2),
        "shannon_entropy": round(entropy_val, 4)
    }


def save_histogram_plot(original_gray: np.ndarray, enhanced_gray: np.ndarray, output_path: str) -> None:
    """Generates and saves a comparative histogram plot comparing raw vs enhanced distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(original_gray.ravel(), bins=256, range=[0, 256], color="navy", alpha=0.7)
    axes[0].set_title("Original Intensity Histogram")
    axes[0].set_xlabel("Pixel Intensity [0-255]")
    axes[0].set_ylabel("Pixel Count")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].hist(enhanced_gray.ravel(), bins=256, range=[0, 256], color="crimson", alpha=0.7)
    axes[1].set_title("Equalized Intensity Histogram")
    axes[1].set_xlabel("Pixel Intensity [0-255]")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
