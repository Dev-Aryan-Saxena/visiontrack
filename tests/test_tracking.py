"""Tests for object tracking and motion parameter estimation."""

import pytest
from src.detection import Detection
from src.tracking import MultiObjectTracker, compute_iou


def test_iou_calculation():
    box_a = (10, 10, 50, 50)
    box_b = (10, 10, 50, 50)
    # Perfect overlap
    assert compute_iou(box_a, box_b) == pytest.approx(1.0)

    # Disjoint boxes
    box_c = (100, 100, 150, 150)
    assert compute_iou(box_a, box_c) == pytest.approx(0.0)

    # Partial overlap
    box_d = (30, 30, 70, 70)
    iou = compute_iou(box_a, box_d)
    assert 0.0 < iou < 1.0


def test_multi_object_tracking_lifecycle():
    tracker = MultiObjectTracker(max_disappeared=3, max_distance=50.0)

    # Frame 1: Register two cars
    det1 = Detection(box=(20, 20, 60, 60), confidence=0.9, class_id=2, class_name="car")
    det2 = Detection(box=(200, 200, 250, 250), confidence=0.85, class_id=2, class_name="car")
    objects = tracker.update([det1, det2], frame_idx=1)

    assert len(objects) == 2
    id1 = objects[0].object_id
    id2 = objects[1].object_id
    assert id1 != id2

    # Frame 2: Objects move slightly (simulate motion)
    det1_moved = Detection(box=(24, 23, 64, 63), confidence=0.9, class_id=2, class_name="car")
    det2_moved = Detection(box=(205, 202, 255, 252), confidence=0.85, class_id=2, class_name="car")
    objects = tracker.update([det1_moved, det2_moved], frame_idx=2)

    assert len(objects) == 2
    # Ensure IDs persisted across frames
    active_ids = {obj.object_id for obj in objects}
    assert id1 in active_ids
    assert id2 in active_ids

    # Verify motion displacement calculated accurately
    obj1 = [o for o in objects if o.object_id == id1][0]
    assert obj1.dx == pytest.approx(4.0, abs=0.5)
    assert obj1.dy == pytest.approx(3.0, abs=0.5)
    assert obj1.instantaneous_speed == pytest.approx(5.0, abs=0.5)  # sqrt(4^2 + 3^2) = 5
    assert obj1.cumulative_distance > 0.0

    # Frame 3-6: One object disappears, verify max_disappeared deregistration
    for f in range(3, 7):
        tracker.update([det1_moved], frame_idx=f)

    # Object 2 should be deregistered after exceeding max_disappeared=3
    assert id2 not in tracker.objects
    assert id1 in tracker.objects
