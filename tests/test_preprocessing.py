"""Tests for image preprocessing, enhancement, and frequency-domain operations."""

import pytest
import numpy as np
from src.preprocessing import (
    to_grayscale, apply_gaussian_filter, apply_median_filter,
    apply_bilateral_filter, resize_image, add_salt_and_pepper_noise
)
from src.enhancement import (
    equalize_histogram_gray, equalize_histogram_color,
    apply_clahe, compute_histogram_metrics
)
from src.frequency import (
    compute_fft, apply_frequency_filter
)


@pytest.fixture
def sample_bgr_image():
    """Generates a synthetic 100x100 BGR color test image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = [255, 0, 0]      # Blue quadrant
    img[:50, 50:] = [0, 255, 0]      # Green quadrant
    img[50:, :50] = [0, 0, 255]      # Red quadrant
    img[50:, 50:] = [200, 200, 200]  # Gray quadrant
    return img


@pytest.fixture
def sample_gray_image(sample_bgr_image):
    return to_grayscale(sample_bgr_image)


def test_grayscale_conversion(sample_bgr_image, sample_gray_image):
    assert sample_gray_image.shape == (100, 100)
    assert sample_gray_image.dtype == np.uint8
    # Test idempotence on already grayscale image
    re_gray = to_grayscale(sample_gray_image)
    assert np.array_equal(sample_gray_image, re_gray)


def test_gaussian_filtering(sample_gray_image):
    filtered = apply_gaussian_filter(sample_gray_image, kernel_size=5, sigma=1.0)
    assert filtered.shape == sample_gray_image.shape
    assert filtered.dtype == np.uint8

    with pytest.raises(ValueError):
        apply_gaussian_filter(sample_gray_image, kernel_size=4)  # Even kernel should fail


def test_median_filtering(sample_gray_image):
    noisy = add_salt_and_pepper_noise(sample_gray_image, salt_prob=0.05, pepper_prob=0.05)
    denoised = apply_median_filter(noisy, kernel_size=5)
    assert denoised.shape == sample_gray_image.shape
    # Denoising should reduce extreme outliers
    assert np.sum(noisy == 255) >= np.sum(denoised == 255)


def test_histogram_equalization(sample_gray_image, sample_bgr_image):
    eq_gray = equalize_histogram_gray(sample_gray_image)
    assert eq_gray.shape == sample_gray_image.shape
    assert eq_gray.dtype == np.uint8

    eq_color = equalize_histogram_color(sample_bgr_image)
    assert eq_color.shape == sample_bgr_image.shape
    assert eq_color.dtype == np.uint8

    clahe_out = apply_clahe(sample_gray_image)
    assert clahe_out.shape == sample_gray_image.shape

    metrics = compute_histogram_metrics(sample_gray_image)
    assert "mean_intensity" in metrics
    assert "shannon_entropy" in metrics


def test_frequency_domain_fft(sample_gray_image):
    f_shift, mag_spec, phase = compute_fft(sample_gray_image)
    assert f_shift.shape == sample_gray_image.shape
    assert mag_spec.shape == sample_gray_image.shape
    assert phase.shape == sample_gray_image.shape

    reconstructed, _ = apply_frequency_filter(sample_gray_image, cutoff_d0=30.0, filter_type="gaussian_lowpass")
    assert reconstructed.shape == sample_gray_image.shape
    assert reconstructed.dtype == np.uint8


def test_resize_image(sample_bgr_image):
    resized = resize_image(sample_bgr_image, target_width=50)
    assert resized.shape[1] == 50
    assert resized.shape[0] == 50
