"""Road-scene object detection engines.

Implements road-scene object detection supporting multiple backends:
1. 'auto' / 'yolo': Ultralytics YOLOv8 lightweight detector (CPU-optimized for person, car, motorcycle, bus, truck).
2. 'hog_svm': Built-in HOG + Linear SVM detector for pedestrians.
3. 'motion': Background-subtraction and morphological moving-blob contour analysis.
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import cv2
import numpy as np

# Road scene classes in standard COCO
ROAD_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


class Detection:
    """Structured representation of a single detected object in an image/frame."""

    def __init__(self, box: Tuple[int, int, int, int], confidence: float,
                 class_id: int, class_name: str) -> None:
        """Args:

        box: (x1, y1, x2, y2) bounding box in pixel coordinates.
        confidence: detection confidence score [0.0, 1.0].
        class_id: category identifier.
        class_name: human-readable category name.
        """
        self.box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.class_name = str(class_name)

    @property
    def center(self) -> Tuple[int, int]:
        """Calculates centroid (cx, cy) of the bounding box."""
        x1, y1, x2, y2 = self.box
        return int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)

    @property
    def width(self) -> int:
        return max(0, self.box[2] - self.box[0])

    @property
    def height(self) -> int:
        return max(0, self.box[3] - self.box[1])

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": list(self.box),
            "center": list(self.center),
            "confidence": round(self.confidence, 3),
            "class_id": self.class_id,
            "class_name": self.class_name
        }


class ObjectDetector:
    """Unified detector interface supporting YOLO, HOG+SVM, and Classical Motion-Blob Detection."""

    def __init__(self, method: str = "auto", confidence_thresh: float = 0.35,
                 model_path: Optional[str] = None) -> None:
        """Args:

        method: 'auto', 'yolo', 'hog_svm', or 'motion'.
        confidence_thresh: minimum detection confidence threshold.
        model_path: optional path to weights file.
        """
        self.requested_method = method.lower()
        self.confidence_thresh = confidence_thresh
        self.model_path = model_path
        self.active_method = "motion"
        self.yolo_model = None
        self.hog_detector = None

        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Initializes the requested detection backend with robust fallback handling."""
        # Setup OpenCV HOG+SVM descriptor if available in this OpenCV build
        if hasattr(cv2, "HOGDescriptor"):
            try:
                self.hog_detector = cv2.HOGDescriptor()
                if hasattr(cv2, "HOGDescriptor_getDefaultPeopleDetector"):
                    self.hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            except Exception:
                self.hog_detector = None
        else:
            self.hog_detector = None

        if self.requested_method in ("yolo", "auto"):
            try:
                from ultralytics import YOLO
                target_weights = self.model_path if self.model_path else "yolov8n.pt"
                print(f"[Detector] Initializing YOLOv8 detector ({target_weights})...")
                self.yolo_model = YOLO(target_weights)
                self.active_method = "yolo"
                print("[Detector] YOLOv8 initialized successfully.")
                return
            except Exception as e:
                if self.requested_method == "yolo":
                    print(f"[Detector Warning] Could not initialize YOLOv8: {e}")
                    print("[Detector Notice] Falling back to Classical Motion-Blob & HOG-SVM detector.")
                self.active_method = "motion"
        elif self.requested_method == "hog_svm":
            self.active_method = "hog_svm"
        else:
            self.active_method = "motion"

        print(f"[Detector] Active detection engine: {self.active_method}")

    def detect(self, frame: np.ndarray,
               motion_regions: Optional[List[Dict[str, Any]]] = None) -> List[Detection]:
        """Detects objects in the input frame using the active backend."""
        if self.active_method == "yolo" and self.yolo_model is not None:
            dets = self._detect_yolo(frame)
            # If auto mode was requested and YOLO found nothing (e.g. in synthetic or non-COCO scenes),
            # gracefully fall back to classical motion-blob detection
            if len(dets) > 0 or self.requested_method == "yolo":
                return dets
            return self._detect_motion_blob(frame, motion_regions)
        elif self.active_method == "hog_svm":
            return self._detect_hog_svm(frame)
        else:
            return self._detect_motion_blob(frame, motion_regions)

    def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        """Inference with YOLOv8."""
        detections: List[Detection] = []
        results = self.yolo_model(frame, conf=self.confidence_thresh, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                # Filter specifically for road scene categories
                if cls_id in ROAD_CLASSES:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0].item())
                    detections.append(Detection(
                        box=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=ROAD_CLASSES[cls_id]
                    ))
        return detections

    def _detect_hog_svm(self, frame: np.ndarray) -> List[Detection]:
        """Inference with HOG + Linear SVM detector."""
        detections: List[Detection] = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Multi-scale HOG detection
        rects, weights = self.hog_detector.detectMultiScale(
            gray, winStride=(4, 4), padding=(8, 8), scale=1.05
        )

        for (x, y, w, h), weight in zip(rects, weights):
            conf = float(weight[0]) if hasattr(weight, "__iter__") else float(weight)
            # Normalize HOG SVM response
            norm_conf = min(1.0, max(0.1, 1.0 / (1.0 + np.exp(-conf))))
            if norm_conf >= self.confidence_thresh:
                detections.append(Detection(
                    box=(int(x), int(y), int(x + w), int(y + h)),
                    confidence=norm_conf,
                    class_id=0,
                    class_name="person"
                ))
        return detections

    def _detect_motion_blob(self, frame: np.ndarray,
                            motion_regions: Optional[List[Dict[str, Any]]] = None) -> List[Detection]:
        """Classical object detection using moving contour bounding boxes and geometry classification."""
        detections: List[Detection] = []
        if not motion_regions:
            return detections

        for region in motion_regions:
            x, y, w, h = region["bbox"]
            area = region["area"]
            aspect_ratio = float(w) / (h + 1e-6)

            # Heuristic geometric classification based on road perspective
            if area > 12000 or (w > 120 and h > 80):
                cls_name = "truck" if aspect_ratio > 1.4 else "bus"
                cls_id = 7 if cls_name == "truck" else 5
                conf = 0.82
            elif area > 2500:
                cls_name = "car"
                cls_id = 2
                conf = 0.88
            elif aspect_ratio < 0.6 and h > 40:
                cls_name = "person"
                cls_id = 0
                conf = 0.78
            else:
                cls_name = "vehicle"
                cls_id = 2
                conf = 0.70

            detections.append(Detection(
                box=(int(x), int(y), int(x + w), int(y + h)),
                confidence=conf,
                class_id=cls_id,
                class_name=cls_name
            ))

        return detections
