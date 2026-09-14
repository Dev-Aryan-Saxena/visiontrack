"""Tests for optical flow motion analysis."""

import pytest
import numpy as np
import cv2
from src.optical_flow import OpticalFlowAnalyzer


@pytest.fixture
def moving_frame_pair():
    """Generates two consecutive video frames where a bright textured square moves diagonally."""
    f1 = np.zeros((120, 120, 3), dtype=np.uint8)
    f2 = np.zeros((120, 120, 3), dtype=np.uint8)

    # Frame 1: Textured block at (30, 30)
    for i in range(30, 60, 4):
        f1[i:i+2, 30:60] = 255

    # Frame 2: Shifted by dx = 4, dy = 3 (motion magnitude = 5.0)
    for i in range(33, 63, 4):
        f2[i:i+2, 34:64] = 255

    return f1, f2


def test_sparse_lucas_kanade_flow(moving_frame_pair):
    f1, f2 = moving_frame_pair
    analyzer = OpticalFlowAnalyzer(method="sparse", max_corners=100)

    # Frame 1 initialization
    vis1, mag1, _ = analyzer.process_frame(f1)
    assert vis1.shape == f1.shape
    assert mag1 == 0.0  # First frame has no motion relative to prior

    # Frame 2 motion computation
    vis2, mag2, metrics = analyzer.process_frame(f2)
    assert vis2.shape == f2.shape
    assert "num_points" in metrics
    assert "avg_magnitude" in metrics


def test_dense_farneback_flow(moving_frame_pair):
    f1, f2 = moving_frame_pair
    analyzer = OpticalFlowAnalyzer(method="dense")

    # Frame 1
    analyzer.process_frame(f1)
    # Frame 2
    vis2, mag2, metrics = analyzer.process_frame(f2)

    assert vis2.shape == f2.shape
    # Motion was injected, so Farneback magnitude should be strictly positive
    assert mag2 > 0.0
    assert "avg_magnitude" in metrics
