"""Multi-object tracking and kinematic parameter estimation.

Implements a multi-object tracker combining:
1. Centroid Euclidean distance matching.
2. Intersection-over-Union (IoU) spatial overlap association.
3. Persistent track lifecycle management (registration, active tracking, occlusion grace period, deregistration).
4. Frame-to-frame motion parameter estimation (dx, dy, instantaneous velocity, cumulative displacement).
"""

from typing import List, Dict, Tuple, Optional, Any
from collections import OrderedDict
import numpy as np
from src.detection import Detection


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes (x1, y1, x2, y2).

    Algorithm Explanation:
    1. What it does: Quantifies the spatial overlap between two rectangular regions.
    2. Mathematical Basis: IoU = Area(A cap B) / Area(A cup B).
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union_area = float(area_a + area_b - inter_area)

    return inter_area / union_area if union_area > 0 else 0.0


class TrackedObject:
    """Represents a single persistent tracked entity throughout video frames."""

    def __init__(self, object_id: int, detection: Detection, frame_idx: int) -> None:
        self.object_id = object_id
        self.class_name = detection.class_name
        self.class_id = detection.class_id
        self.box = detection.box
        self.center = detection.center
        self.start_frame = frame_idx
        self.last_frame = frame_idx
        self.disappeared_count = 0

        # Motion metrics
        self.dx = 0.0
        self.dy = 0.0
        self.instantaneous_speed = 0.0  # pixels/frame
        self.cumulative_distance = 0.0  # total pixels traveled

        # Trajectory history: list of (x, y) centers for visual trail
        self.trajectory: List[Tuple[int, int]] = [self.center]

    def update(self, detection: Detection, frame_idx: int) -> None:
        """Updates track state with a new matched detection."""
        prev_center = self.center
        self.box = detection.box
        self.center = detection.center
        self.last_frame = frame_idx
        self.disappeared_count = 0

        # Calculate motion parameters
        self.dx = float(self.center[0] - prev_center[0])
        self.dy = float(self.center[1] - prev_center[1])
        self.instantaneous_speed = float(np.sqrt(self.dx ** 2 + self.dy ** 2))
        self.cumulative_distance += self.instantaneous_speed

        self.trajectory.append(self.center)
        if len(self.trajectory) > 40:
            self.trajectory.pop(0)

    def mark_missed(self) -> None:
        """Increments missed frames counter during temporary occlusion."""
        self.disappeared_count += 1
        self.dx = 0.0
        self.dy = 0.0
        self.instantaneous_speed = 0.0


class MultiObjectTracker:
    """Manages tracking multiple objects across consecutive video frames."""

    def __init__(self, max_disappeared: int = 15, max_distance: float = 85.0,
                 min_iou: float = 0.20) -> None:
        """Args:

        max_disappeared: Number of consecutive frames an object can be lost before deregistering.
        max_distance: Maximum allowable Euclidean centroid pixel distance for association.
        min_iou: Minimum IoU overlap when using hybrid matching.
        """
        self.next_object_id = 1
        self.objects: OrderedDict[int, TrackedObject] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_iou = min_iou

    def register(self, detection: Detection, frame_idx: int) -> None:
        """Registers a new tracked object."""
        self.objects[self.next_object_id] = TrackedObject(self.next_object_id, detection, frame_idx)
        self.next_object_id += 1

    def deregister(self, object_id: int) -> None:
        """Removes a lost object from the active tracking pool."""
        if object_id in self.objects:
            del self.objects[object_id]

    def update(self, detections: List[Detection], frame_idx: int) -> List[TrackedObject]:
        """Matches current frame detections with existing tracked objects.

        Algorithm Explanation:
        1. If no existing objects, register all incoming detections.
        2. If incoming detections empty, mark all existing objects as missed.
        3. Form distance matrix between existing centroids and incoming detection centroids.
        4. Match greedily by minimum distance, with secondary verification via IoU overlap.
        5. Unmatched detections are registered as new entities.
        6. Unmatched existing objects have missed counter incremented.
        """
        # Case 1: No detections this frame
        if len(detections) == 0:
            lost_ids = []
            for obj_id, obj in self.objects.items():
                obj.mark_missed()
                if obj.disappeared_count > self.max_disappeared:
                    lost_ids.append(obj_id)
            for obj_id in lost_ids:
                self.deregister(obj_id)
            return list(self.objects.values())

        # Case 2: No tracked objects currently active
        if len(self.objects) == 0:
            for det in detections:
                self.register(det, frame_idx)
            return list(self.objects.values())

        # Case 3: Match existing objects to current detections
        object_ids = list(self.objects.keys())
        object_centers = np.array([self.objects[oid].center for oid in object_ids])
        detection_centers = np.array([det.center for det in detections])

        # Compute Euclidean distance matrix (N_objects x M_detections)
        diff = object_centers[:, np.newaxis, :] - detection_centers[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)

        # Greedy association by minimum distance
        rows = dist_matrix.min(axis=1).argsort()
        cols = dist_matrix.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            dist = dist_matrix[row, col]
            obj_id = object_ids[row]
            det = detections[col]

            # IoU check for bounding box continuity
            iou = compute_iou(self.objects[obj_id].box, det.box)

            # Associate if within distance threshold or has reasonable IoU overlap
            if dist <= self.max_distance or iou >= self.min_iou:
                self.objects[obj_id].update(det, frame_idx)
                used_rows.add(row)
                used_cols.add(col)

        # Check for unassociated existing objects
        unused_rows = set(range(len(object_ids))) - used_rows
        lost_ids = []
        for row in unused_rows:
            obj_id = object_ids[row]
            self.objects[obj_id].mark_missed()
            if self.objects[obj_id].disappeared_count > self.max_disappeared:
                lost_ids.append(obj_id)

        for obj_id in lost_ids:
            self.deregister(obj_id)

        # Register new objects from unassociated detections
        unused_cols = set(range(len(detections))) - used_cols
        for col in unused_cols:
            self.register(detections[col], frame_idx)

        return list(self.objects.values())
