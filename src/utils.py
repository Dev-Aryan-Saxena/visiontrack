"""Module: utils.py
Computer Vision Syllabus Mapping:
- General Utilities, Synthetic Benchmark Generation, Video I/O,
  Quantitative Evaluation, Headless Data Serialization (CSV/JSON).

Description:
Provides essential system utilities:
1. Timestamped output directory management.
2. Headless video reader/writer wrappers with multi-codec fallbacks.
3. Built-in synthetic test data generator (road images, traffic video, stereo pairs)
   ensuring 100% self-contained reproducibility.
4. Annotated visualization HUD overlays (bounding boxes, track IDs, velocity vectors).
5. Machine-readable tabular CSV and structured JSON exporters.
"""

from typing import Tuple, List, Dict, Any, Optional
import os
import datetime
import json
import csv
import cv2
import numpy as np


def create_experiment_dir(base_output_dir: str = "outputs", mode: str = "video") -> str:
    """Creates a unique timestamped experiment directory to prevent overwriting prior runs."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_output_dir, mode, f"exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def create_video_writer(output_path: str, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    """Creates a headless-safe VideoWriter attempting robust cross-platform codecs."""
    w, h = frame_size
    codecs_to_try = ["mp4v", "avc1", "XVID", "MJPG"]

    for codec in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=True)
        if writer.isOpened():
            return writer

    # Fallback to default
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=True)


def draw_annotated_frame(frame: np.ndarray, tracked_objects: List[Any],
                         frame_idx: int, total_frames: int,
                         avg_motion_speed: float,
                         optical_flow_mag: float) -> np.ndarray:
    """Draws professional academic bounding boxes, IDs, trajectories, and HUD metrics."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Color palette for classes
    color_map = {
        "car": (0, 165, 255),       # Orange
        "truck": (0, 215, 255),     # Amber
        "bus": (255, 144, 30),      # Blue/Teal
        "person": (0, 255, 0),      # Bright Green
        "motorcycle": (255, 0, 255),# Magenta
        "vehicle": (0, 200, 200)    # Yellow
    }

    # 1. Draw trajectory history trails
    for obj in tracked_objects:
        trail = obj.trajectory
        if len(trail) > 1:
            for i in range(1, len(trail)):
                thickness = int(np.sqrt(float(i + 1)) * 1.2)
                cv2.line(vis, trail[i - 1], trail[i], (0, 255, 255), thickness, cv2.LINE_AA)

    # 2. Draw bounding boxes and labels
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.box
        color = color_map.get(obj.class_name.lower(), (0, 255, 0))

        # Main bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Center dot
        cx, cy = obj.center
        cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

        # Label banner: e.g. "Car #1 | 4.2 px/f"
        label = f"{obj.class_name.capitalize()} #{obj.object_id} | {obj.instantaneous_speed:.1f} px/f"
        (lbl_w, lbl_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        top_y = max(lbl_h + 6, y1)
        cv2.rectangle(vis, (x1, top_y - lbl_h - 6), (x1 + lbl_w + 6, top_y), color, -1)
        cv2.putText(vis, label, (x1 + 3, top_y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # 3. Academic HUD Overlay Header
    hud_overlay = vis.copy()
    cv2.rectangle(hud_overlay, (10, 10), (340, 110), (20, 20, 20), -1)
    cv2.addWeighted(hud_overlay, 0.75, vis, 0.25, 0, vis)
    cv2.rectangle(vis, (10, 10), (340, 110), (100, 100, 100), 1)

    cv2.putText(vis, "VisionTrack Scene Analysis", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"Frame: {frame_idx}/{total_frames if total_frames > 0 else 'N/A'}",
                (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.putText(vis, f"Tracked Entities: {len(tracked_objects)}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.putText(vis, f"Mean Motion Rate: {avg_motion_speed:.2f} px/frame", (20, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)
    cv2.putText(vis, f"Optical Flow Mag: {optical_flow_mag:.2f}", (20, 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)

    return vis


def save_tracking_csv(tracking_records: List[Dict[str, Any]], output_csv_path: str) -> None:
    """Exports frame-by-frame tracking records to structured tabular CSV."""
    if not tracking_records:
        return

    fieldnames = [
        "frame_idx", "object_id", "class_name",
        "center_x", "center_y",
        "box_x1", "box_y1", "box_x2", "box_y2",
        "displacement_dx", "displacement_dy",
        "instantaneous_speed_px_per_frame",
        "cumulative_distance_px"
    ]

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in tracking_records:
            writer.writerow(r)


def save_summary_json(summary_data: Dict[str, Any], output_json_path: str) -> None:
    """Exports complete quantitative experiment summary to structured machine-readable JSON."""
    with open(output_json_path, mode="w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)


# =====================================================================
# Synthetic Road Scene Generator (100% Reproducible Test Media)
# =====================================================================

def generate_synthetic_road_image(output_path: str, width: int = 640, height: int = 360) -> str:
    """Synthesizes a realistic road scene image with road surface, lanes, sky, and vehicles."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Sky (gradient blue)
    for y in range(int(height * 0.45)):
        ratio = y / (height * 0.45)
        blue = int(180 + 60 * (1 - ratio))
        green = int(140 + 50 * (1 - ratio))
        red = int(90 + 30 * (1 - ratio))
        img[y, :] = [blue, green, red]

    # 2. Road surface (asphalt gray)
    horizon_y = int(height * 0.45)
    img[horizon_y:, :] = [75, 75, 75]

    # Road perspective polygon
    road_pts = np.array([
        [int(width * 0.40), horizon_y],
        [int(width * 0.60), horizon_y],
        [width, height],
        [0, height]
    ], dtype=np.int32)
    cv2.fillPoly(img, [road_pts], (60, 60, 60))

    # Curbs/Grass roadside
    left_grass = np.array([[0, horizon_y], [int(width * 0.40), horizon_y], [0, height]], dtype=np.int32)
    right_grass = np.array([[int(width * 0.60), horizon_y], [width, horizon_y], [width, height]], dtype=np.int32)
    cv2.fillPoly(img, [left_grass], (34, 120, 34))
    cv2.fillPoly(img, [right_grass], (34, 120, 34))

    # 3. Yellow centerline lane markings
    for t in np.linspace(0.1, 1.0, 7):
        y1 = int(horizon_y + (height - horizon_y) * (t - 0.05))
        y2 = int(horizon_y + (height - horizon_y) * t)
        cx = int(width * 0.50)
        thick = max(2, int(6 * t))
        cv2.line(img, (cx, y1), (cx, y2), (0, 215, 255), thick)

    # White lane divider markings (left and right lanes)
    for t in np.linspace(0.15, 0.95, 6):
        y1 = int(horizon_y + (height - horizon_y) * (t - 0.04))
        y2 = int(horizon_y + (height - horizon_y) * t)
        w_left = int(width * 0.45 - (width * 0.25) * t)
        w_right = int(width * 0.55 + (width * 0.25) * t)
        cv2.line(img, (w_left, y1), (w_left, y2), (240, 240, 240), max(1, int(4 * t)))
        cv2.line(img, (w_right, y1), (w_right, y2), (240, 240, 240), max(1, int(4 * t)))

    # 4. Vehicles
    # Car 1 (Left lane, red car)
    c1_x, c1_y = int(width * 0.28), int(height * 0.70)
    cv2.rectangle(img, (c1_x, c1_y), (c1_x + 90, c1_y + 45), (20, 30, 180), -1)
    cv2.rectangle(img, (c1_x + 15, c1_y - 20), (c1_x + 75, c1_y), (30, 40, 200), -1)
    cv2.rectangle(img, (c1_x + 20, c1_y - 16), (c1_x + 70, c1_y - 2), (180, 200, 220), -1)  # Windshield
    # Wheels
    cv2.circle(img, (c1_x + 20, c1_y + 45), 10, (20, 20, 20), -1)
    cv2.circle(img, (c1_x + 70, c1_y + 45), 10, (20, 20, 20), -1)

    # Car 2 (Right lane, silver sedan)
    c2_x, c2_y = int(width * 0.62), int(height * 0.58)
    cv2.rectangle(img, (c2_x, c2_y), (c2_x + 65, c2_y + 35), (170, 170, 170), -1)
    cv2.rectangle(img, (c2_x + 12, c2_y - 15), (c2_x + 55, c2_y), (190, 190, 190), -1)
    cv2.rectangle(img, (c2_x + 16, c2_y - 12), (c2_x + 50, c2_y - 2), (210, 220, 230), -1)
    cv2.circle(img, (c2_x + 16, c2_y + 35), 8, (20, 20, 20), -1)
    cv2.circle(img, (c2_x + 50, c2_y + 35), 8, (20, 20, 20), -1)

    # Save to disk
    cv2.imwrite(output_path, img)
    return output_path


def generate_synthetic_traffic_video(output_path: str, num_frames: int = 120,
                                     fps: int = 30, width: int = 640, height: int = 360) -> str:
    """Synthesizes a 120-frame traffic video featuring moving vehicles with known velocity profiles."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = create_video_writer(output_path, float(fps), (width, height))

    horizon_y = int(height * 0.45)

    # Vehicle state definitions: [initial_x, initial_y, speed_x, speed_y, width, height, color, class_name]
    v1 = {"x": float(width * 0.25), "y": float(height * 0.88), "vx": 0.0, "vy": -2.2, "w": 85, "h": 50, "col": (20, 40, 200), "type": "car"}
    v2 = {"x": float(width * 0.65), "y": float(height * 0.52), "vx": 0.0, "vy": 3.4, "w": 75, "h": 45, "col": (200, 120, 20), "type": "car"}
    v3 = {"x": float(width * 0.80), "y": float(height * 0.60), "vx": 0.0, "vy": 1.8, "w": 110, "h": 70, "col": (50, 150, 50), "type": "truck"}

    for f in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 1. Sky
        for y in range(horizon_y):
            ratio = y / horizon_y
            frame[y, :] = [int(180 + 50 * (1 - ratio)), int(140 + 40 * (1 - ratio)), int(90 + 30 * (1 - ratio))]

        # 2. Road & Grass
        frame[horizon_y:, :] = [70, 70, 70]
        road_pts = np.array([[int(width * 0.38), horizon_y], [int(width * 0.62), horizon_y],
                             [width, height], [0, height]], dtype=np.int32)
        cv2.fillPoly(frame, [road_pts], (55, 55, 55))
        left_grass = np.array([[0, horizon_y], [int(width * 0.38), horizon_y], [0, height]], dtype=np.int32)
        right_grass = np.array([[int(width * 0.62), horizon_y], [width, horizon_y], [width, height]], dtype=np.int32)
        cv2.fillPoly(frame, [left_grass], (30, 110, 30))
        cv2.fillPoly(frame, [right_grass], (30, 110, 30))

        # Lane markings with motion animation
        offset = (f * 4) % 40
        for t in np.linspace(0.05, 1.0, 8):
            y_base = horizon_y + (height - horizon_y) * t + (offset * t)
            y1 = int(min(height - 1, y_base))
            y2 = int(min(height - 1, y_base + 14 * t))
            if y2 > y1 and y1 >= horizon_y:
                cv2.line(frame, (width // 2, y1), (width // 2, y2), (0, 215, 255), max(2, int(5 * t)))

        # Update vehicle positions
        v1["y"] += v1["vy"]
        v2["y"] += v2["vy"]
        v3["y"] += v3["vy"]

        # Loop vehicles if out of bounds
        if v1["y"] < horizon_y + 10:
            v1["y"] = height + 10
        if v2["y"] > height + 20:
            v2["y"] = horizon_y + 10
        if v3["y"] > height + 30:
            v3["y"] = horizon_y + 15

        # Render vehicles
        for v in [v1, v2, v3]:
            # Scale vehicle size slightly with perspective perspective (y coordinate)
            scale = max(0.5, (v["y"] - horizon_y) / (height - horizon_y))
            vw = int(v["w"] * scale)
            vh = int(v["h"] * scale)
            vx = int(v["x"] - vw / 2.0)
            vy = int(v["y"] - vh / 2.0)

            if horizon_y <= vy <= height + 50:
                cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), v["col"], -1)
                # Windshield
                ww = max(4, int(vw * 0.6))
                wh = max(3, int(vh * 0.35))
                cv2.rectangle(frame, (vx + (vw - ww) // 2, vy + int(vh * 0.1)),
                              (vx + (vw - ww) // 2 + ww, vy + int(vh * 0.1) + wh), (200, 220, 240), -1)

        writer.write(frame)

    writer.release()
    return output_path


def generate_synthetic_stereo_pair(left_path: str, right_path: str,
                                   width: int = 640, height: int = 360) -> Tuple[str, str]:
    """Synthesizes a rectified binocular stereo image pair demonstrating horizontal disparity."""
    os.makedirs(os.path.dirname(left_path), exist_ok=True)
    os.makedirs(os.path.dirname(right_path), exist_ok=True)

    # Base road scene
    left_img = np.zeros((height, width, 3), dtype=np.uint8)
    right_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Background horizon (distant = zero disparity)
    for y in range(height):
        ratio = y / height
        col = [int(100 + 80 * ratio), int(100 + 80 * ratio), int(100 + 80 * ratio)]
        left_img[y, :] = col
        right_img[y, :] = col

    # Ground plane texture
    np.random.seed(42)
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    left_img = cv2.add(left_img, noise)
    right_img = cv2.add(right_img, noise)

    # Object 1: Close car (foreground -> large disparity: d = 28 pixels)
    disp1 = 28
    c1_w, c1_h = 130, 80
    c1_y = int(height * 0.55)
    c1_x_left = int(width * 0.25)
    c1_x_right = c1_x_left - disp1

    cv2.rectangle(left_img, (c1_x_left, c1_y), (c1_x_left + c1_w, c1_y + c1_h), (20, 40, 220), -1)
    cv2.rectangle(right_img, (c1_x_right, c1_y), (c1_x_right + c1_w, c1_y + c1_h), (20, 40, 220), -1)

    # Object 2: Midground vehicle (d = 16 pixels)
    disp2 = 16
    c2_w, c2_h = 80, 50
    c2_y = int(height * 0.35)
    c2_x_left = int(width * 0.65)
    c2_x_right = c2_x_left - disp2

    cv2.rectangle(left_img, (c2_x_left, c2_y), (c2_x_left + c2_w, c2_y + c2_h), (220, 140, 20), -1)
    cv2.rectangle(right_img, (c2_x_right, c2_y), (c2_x_right + c2_w, c2_y + c2_h), (220, 140, 20), -1)

    cv2.imwrite(left_path, left_img)
    cv2.imwrite(right_path, right_img)
    return left_path, right_path
