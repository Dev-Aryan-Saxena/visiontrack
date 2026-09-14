"""Frequency-domain 2D Fourier analysis and filtering.

Implements 2D Discrete Fourier Transform (DFT), magnitude and phase spectrum extraction,
ideal/Gaussian/Butterworth low-pass and high-pass filtering in the frequency domain,
and spatial reconstruction via the Inverse Discrete Fourier Transform (IDFT).
"""

from typing import Tuple, Literal
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_fft(gray_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the 2D Fast Fourier Transform (FFT) and centered spectra.

    Algorithm Explanation:
    1. What it does: Decomposes a spatial image into its constituent orthogonal sinusoidal frequencies.
    2. Why it is used: Fundamental tool for frequency analysis, revealing periodic noise, directional patterns,
       and frequency power distribution. By the Convolution Theorem, spatial convolution f * h is equivalent
       to pointwise multiplication F(u,v) * H(u,v) in the frequency domain.
    3. Mathematical Basis: F(u, v) = sum_{x=0}^{M-1} sum_{y=0}^{N-1} f(x, y) * exp(-j * 2 * pi * (u*x/M + v*y/N)).
       Magnitude S(u,v) = log(1 + |F(u,v)|) scales the immense dynamic range for human visual analysis.
    4. Input: Grayscale image (H x W, uint8).
    5. Output: Tuple of (f_shift complex array, magnitude_spectrum uint8, phase_spectrum float32).
    """
    if len(gray_image.shape) != 2:
        raise ValueError("Input must be a single-channel grayscale image.")

    # Convert to float32 for high precision DFT
    f = np.fft.fft2(gray_image.astype(np.float32))
    f_shift = np.fft.fftshift(f)

    # Magnitude spectrum in log scale: 20 * log(1 + |F|)
    magnitude = np.abs(f_shift)
    magnitude_spectrum = 20 * np.log(magnitude + 1.0)
    # Normalize to 0-255 uint8 for visualization
    norm_mag = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Phase spectrum
    phase = np.angle(f_shift)

    return f_shift, norm_mag, phase


def create_frequency_filter(shape: Tuple[int, int], cutoff_d0: float = 30.0,
                            filter_type: Literal["gaussian_lowpass", "gaussian_highpass",
                                                 "ideal_lowpass", "ideal_highpass",
                                                 "butterworth_lowpass"] = "gaussian_lowpass",
                            order: int = 2) -> np.ndarray:
    """Generates centered 2D frequency domain transfer functions H(u, v).

    Algorithm Explanation:
    1. Ideal Filter: Sharp discontinuity at cutoff radius D0. Produces spatial ringing (Gibbs phenomenon).
    2. Gaussian Filter: H(u, v) = exp(-D^2(u, v) / (2 * D0^2)). Perfectly smooth without Gibbs ringing.
    3. Butterworth Filter: H(u, v) = 1 / (1 + (D(u, v) / D0)^(2 * order)). Transition smoothness controlled by order n.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    u_grid, v_grid = np.meshgrid(u, v, indexing="ij")
    d_uv = np.sqrt(u_grid ** 2 + v_grid ** 2)

    if filter_type == "ideal_lowpass":
        h = (d_uv <= cutoff_d0).astype(np.float32)
    elif filter_type == "ideal_highpass":
        h = (d_uv > cutoff_d0).astype(np.float32)
    elif filter_type == "gaussian_lowpass":
        h = np.exp(-(d_uv ** 2) / (2 * (cutoff_d0 ** 2)))
    elif filter_type == "gaussian_highpass":
        h = 1.0 - np.exp(-(d_uv ** 2) / (2 * (cutoff_d0 ** 2)))
    elif filter_type == "butterworth_lowpass":
        h = 1.0 / (1.0 + (d_uv / (cutoff_d0 + 1e-8)) ** (2 * order))
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return h.astype(np.float32)


def apply_frequency_filter(gray_image: np.ndarray, cutoff_d0: float = 30.0,
                           filter_type: str = "gaussian_lowpass") -> Tuple[np.ndarray, np.ndarray]:
    """Applies frequency domain filtering and reconstructs spatial domain image via IDFT.

    Algorithm Explanation:
    1. Computes centered FFT: F(u, v).
    2. Multiplies by transfer function: G(u, v) = F(u, v) * H(u, v).
    3. Computes inverse FFT: g(x, y) = Real(ifft2(ifftshift(G(u, v)))).
    """
    f_shift, _, _ = compute_fft(gray_image)
    h_filter = create_frequency_filter(gray_image.shape, cutoff_d0=cutoff_d0, filter_type=filter_type)

    # Pointwise multiplication in frequency domain
    g_shift = f_shift * h_filter

    # Inverse Fourier Transform
    g_ishift = np.fft.ifftshift(g_shift)
    img_back = np.fft.ifft2(g_ishift)
    img_back_real = np.real(img_back)

    # Clip to valid pixel range
    reconstructed = np.clip(img_back_real, 0, 255).astype(np.uint8)
    return reconstructed, (h_filter * 255).astype(np.uint8)


def save_fourier_analysis_plot(gray_image: np.ndarray, output_path: str, cutoff_d0: float = 40.0) -> None:
    """Generates a complete 4-panel Fourier Transform visualization."""
    _, mag_spectrum, _ = compute_fft(gray_image)
    lp_reconstructed, lp_filter = apply_frequency_filter(gray_image, cutoff_d0=cutoff_d0, filter_type="gaussian_lowpass")
    hp_reconstructed, hp_filter = apply_frequency_filter(gray_image, cutoff_d0=cutoff_d0, filter_type="gaussian_highpass")

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(gray_image, cmap="gray")
    axes[0, 0].set_title("Input Grayscale Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mag_spectrum, cmap="inferno")
    axes[0, 1].set_title("Centered FFT Magnitude Spectrum (Log)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(lp_reconstructed, cmap="gray")
    axes[1, 0].set_title(f"Reconstructed: Gaussian Low-Pass (D0={cutoff_d0})")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(hp_reconstructed, cmap="gray")
    axes[1, 1].set_title(f"Reconstructed: Gaussian High-Pass (D0={cutoff_d0})")
    axes[1, 1].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
