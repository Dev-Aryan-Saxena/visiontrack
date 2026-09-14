"""Tests for image segmentation algorithms."""

import pytest
import numpy as np
from src.segmentation import (
    segment_by_threshold, segment_by_kmeans,
    segment_by_region_growing, segment_by_mean_shift
)


@pytest.fixture
def bimodal_image():
    """Generates a bimodal image with two clear intensity distributions (30 and 220)."""
    img = np.zeros((80, 80), dtype=np.uint8)
    img[:40, :] = np.clip(np.random.normal(40, 5, (40, 80)), 0, 255).astype(np.uint8)
    img[40:, :] = np.clip(np.random.normal(200, 5, (40, 80)), 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def color_image():
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    img[:40, :] = [255, 0, 0]
    img[40:, :] = [0, 255, 0]
    return img


def test_otsu_thresholding(bimodal_image):
    binary, thresh_val = segment_by_threshold(bimodal_image, method="otsu")
    assert binary.shape == bimodal_image.shape
    assert binary.dtype == np.uint8
    # Optimal threshold for distributions around 40 and 200 should fall cleanly between them
    assert 40 < thresh_val < 200
    # Values must be strictly binary: 0 or 255
    unique_vals = set(np.unique(binary))
    assert unique_vals.issubset({0, 255})


def test_adaptive_thresholding(bimodal_image):
    binary, _ = segment_by_threshold(bimodal_image, method="adaptive")
    assert binary.shape == bimodal_image.shape
    unique_vals = set(np.unique(binary))
    assert unique_vals.issubset({0, 255})


def test_kmeans_segmentation(color_image):
    k = 2
    segmented, labels = segment_by_kmeans(color_image, k=k)
    assert segmented.shape == color_image.shape
    assert labels.shape == color_image.shape[:2]
    # Exactly k unique labels should be present
    assert len(np.unique(labels)) == k


def test_region_growing(bimodal_image):
    # Seed inside the high-intensity region
    mask = segment_by_region_growing(bimodal_image, seeds=[(40, 60)], intensity_threshold=20.0)
    assert mask.shape == bimodal_image.shape
    assert mask.dtype == np.uint8
    # Lower half should be segmented
    assert mask[60, 40] == 255
    # Upper half should remain unsegmented
    assert mask[20, 40] == 0


def test_meanshift_segmentation(color_image):
    filtered = segment_by_mean_shift(color_image, spatial_radius=10, color_radius=20)
    assert filtered.shape == color_image.shape
    assert filtered.dtype == np.uint8
