#!/usr/bin/env python3
"""VisionTrack: Road Scene Analysis and Motion Tracking System.

A command-line Computer Vision application for road scene analysis from images and videos.

Features:
- Image preprocessing, enhancement, edge detection, feature extraction, and segmentation.
- Video analysis with dynamic background subtraction, lightweight object detection,
  multi-object tracking with persistent IDs, and differential optical flow.
- Perspective transformation utility for bird's-eye view road mapping.
- Optional binocular stereo disparity and depth estimation.

Usage:
    python main.py --help
    python main.py --generate-samples
    python main.py --mode image --input data/sample_road.jpg
    python main.py --mode image --input data/sample_road.jpg --perspective
    python main.py --mode video --input data/sample_traffic.mp4
    python main.py --mode video --input data/sample_traffic.mp4 --optical-flow dense
    python main.py --mode stereo --left data/left_stereo.jpg --right data/right_stereo.jpg
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

# VisionTrack modules
from src.preprocessing import (
    to_grayscale, apply_gaussian_filter, apply_median_filter,
    apply_bilateral_filter, resize_image
)
from src.enhancement import (
    equalize_histogram_gray, apply_clahe, compute_histogram_metrics
)
from src.features import (
    detect_canny_edges, detect_hough_lines, detect_harris_corners,
    extract_hog_features
)
from src.segmentation import (
    segment_by_threshold, segment_by_kmeans,
    segment_by_region_growing, segment_by_mean_shift
)
from src.background import BackgroundSubtractor
from src.detection import ObjectDetector
from src.tracking import MultiObjectTracker
from src.optical_flow import OpticalFlowAnalyzer
from src.perspective import PerspectiveTransformer
from src.stereo import StereoDepthEstimator
from src.utils import (
    create_experiment_dir, create_video_writer, draw_annotated_frame,
    save_tracking_csv, save_summary_json,
    generate_synthetic_road_image, generate_synthetic_traffic_video,
    generate_synthetic_stereo_pair
)


def run_image_pipeline(args: argparse.Namespace) -> int:
    """Executes the image analysis pipeline and saves essential visual outputs."""
    input_path = args.input
    if not input_path or not os.path.isfile(input_path):
        print(f"ERROR: Input image file '{input_path}' was not found.")
        return 1

    img_bgr = cv2.imread(input_path)
    if img_bgr is None or img_bgr.size == 0:
        print(f"ERROR: Failed to decode image '{input_path}'. Corrupted or unsupported format.")
        return 1

    if args.resize:
        img_bgr = resize_image(img_bgr, target_width=args.resize)

    h, w = img_bgr.shape[:2]
    out_dir = create_experiment_dir(base_output_dir=args.output_dir, mode="image")
    print(f"\n=======================================================")
    print(f"VisionTrack: Image Scene Analysis")
    print(f"Input: {input_path} ({w}x{h})")
    print(f"Output Directory: {out_dir}")
    print(f"=======================================================\n")

    summary: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "image",
        "input_file": os.path.abspath(input_path),
        "resolution": {"width": w, "height": h},
        "selected_segmentation": args.segment
    }

    # 1. Original Image
    print("[1/5] Loading original input image...")
    cv2.imwrite(os.path.join(out_dir, "original.jpg"), img_bgr)

    # 2. Preprocessing & Enhancement
    print("[2/5] Applying preprocessing and contrast enhancement...")
    gray = to_grayscale(img_bgr)
    smoothed = apply_gaussian_filter(gray, kernel_size=5, sigma=1.2)
    enhanced = apply_clahe(smoothed, clip_limit=2.5)
    cv2.imwrite(os.path.join(out_dir, "filtered.jpg"), enhanced)

    summary["histogram_metrics"] = compute_histogram_metrics(gray)

    # 3. Edge Detection
    print("[3/5] Performing Canny edge detection...")
    edges = detect_canny_edges(smoothed, low_threshold=args.canny_low, high_threshold=args.canny_high)
    cv2.imwrite(os.path.join(out_dir, "edges.jpg"), edges)

    # 4. Feature Detection (Hough lines & Harris corners combined on visual frame)
    print(f"[4/5] Extracting geometric features (Hough lines & Harris corners)...")
    feature_vis = img_bgr.copy()

    # Hough Lines
    _, lines = detect_hough_lines(gray, canny_low=args.canny_low, canny_high=args.canny_high)
    for x1, y1, x2, y2 in lines:
        cv2.line(feature_vis, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

    # Harris Corners
    _, _, num_corners = detect_harris_corners(gray)
    gray_float = np.float32(gray)
    dst = cv2.cornerHarris(gray_float, blockSize=3, ksize=3, k=0.04)
    dst_dilated = cv2.dilate(dst, None)
    thresh = 0.01 * dst.max() if dst.max() > 0 else 0
    corner_mask = (dst > thresh) & (dst == dst_dilated)
    y_coords, x_coords = np.where(corner_mask)
    for cx, cy in zip(x_coords, y_coords):
        cv2.circle(feature_vis, (int(cx), int(cy)), 4, (0, 0, 255), -1, cv2.LINE_AA)

    cv2.imwrite(os.path.join(out_dir, "features.jpg"), feature_vis)
    summary["hough_lines_count"] = len(lines)
    summary["harris_corners_count"] = int(num_corners)

    # 5. Image Segmentation
    print(f"[5/5] Executing image segmentation (Method: {args.segment})...")
    if args.segment == "kmeans":
        segmented_img, _ = segment_by_kmeans(img_bgr, k=args.kmeans_k)
    elif args.segment == "otsu":
        segmented_img, otsu_t = segment_by_threshold(gray, method="otsu")
        summary["otsu_threshold"] = otsu_t
    elif args.segment == "adaptive":
        segmented_img, _ = segment_by_threshold(gray, method="adaptive")
    elif args.segment == "region_growing":
        segmented_img = segment_by_region_growing(gray)
    elif args.segment == "meanshift":
        segmented_img = segment_by_mean_shift(img_bgr)
    else:
        segmented_img, _ = segment_by_kmeans(img_bgr, k=args.kmeans_k)

    cv2.imwrite(os.path.join(out_dir, "segmentation.jpg"), segmented_img)

    # 6. Perspective Transformation (Optional Bird's-Eye View)
    if args.perspective:
        print("[*] Computing bird's-eye view perspective transformation...")
        pt = PerspectiveTransformer()
        bev_img = pt.warp_to_birds_eye(img_bgr)
        cv2.imwrite(os.path.join(out_dir, "birds_eye.jpg"), bev_img)
        summary["birds_eye_view_generated"] = True

    # Save summary JSON
    save_summary_json(summary, os.path.join(out_dir, "summary.json"))
    print(f"\n[VisionTrack] Image analysis completed successfully!")
    print(f"[VisionTrack] Outputs saved to: {out_dir}\n")
    return 0


def run_video_pipeline(args: argparse.Namespace) -> int:
    """Executes the video analysis, tracking, and optical flow pipeline."""
    input_path = args.input
    if not input_path or not os.path.isfile(input_path):
        print(f"ERROR: Input video file '{input_path}' was not found.")
        return 1

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Failed to open video file '{input_path}'. Unsupported codec or corrupted stream.")
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine processing dimensions
    if args.resize:
        proc_w = args.resize
        proc_h = int(round(orig_h * (args.resize / orig_w)))
    else:
        proc_w, proc_h = orig_w, orig_h

    out_dir = create_experiment_dir(base_output_dir=args.output_dir, mode="video")

    print(f"\n=======================================================")
    print(f"VisionTrack: Video Analysis & Tracking")
    print(f"Input: {input_path}")
    print(f"Frames: {total_frames} | FPS: {fps:.2f} | Resolution: {proc_w}x{proc_h}")
    print(f"Detector: {args.detector} | Optical Flow: {args.optical_flow}")
    print(f"Output Directory: {out_dir}")
    print(f"=======================================================\n")

    # Initialize sub-systems
    bg_subtractor = BackgroundSubtractor(
        method=args.bg_method, history=args.bg_history,
        var_threshold=args.bg_var_threshold, detect_shadows=True,
        min_contour_area=args.min_area
    )
    detector = ObjectDetector(
        method=args.detector, confidence_thresh=args.conf_thresh
    )
    tracker = MultiObjectTracker(
        max_disappeared=args.max_disappeared, max_distance=args.max_distance
    )
    flow_analyzer = OpticalFlowAnalyzer(
        method=args.optical_flow
    )

    # Initialize video writers
    annotated_video_path = os.path.join(out_dir, "annotated_video.mp4")
    flow_video_path = os.path.join(out_dir, "optical_flow.mp4")

    annotated_writer = None
    flow_writer = None

    if args.save_video:
        annotated_writer = create_video_writer(annotated_video_path, fps, (proc_w, proc_h))
        flow_writer = create_video_writer(flow_video_path, fps, (proc_w, proc_h))

    tracking_records: List[Dict[str, Any]] = []
    frame_idx = 0
    processed_count = 0
    motion_speeds: List[float] = []
    optical_flow_mags: List[float] = []
    max_frames = args.max_frames if args.max_frames > 0 else total_frames

    print("Processing video...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if args.max_frames > 0 and frame_idx > args.max_frames:
                break

            # Frame skipping
            if frame_idx % args.frame_skip != 0:
                continue

            processed_count += 1
            if args.resize:
                frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

            # 1. Background subtraction & morphological cleaning
            raw_mask, cleaned_mask = bg_subtractor.apply(frame)
            moving_regions = bg_subtractor.extract_moving_regions(cleaned_mask)

            # 2. Object detection
            detections = detector.detect(frame, motion_regions=moving_regions)

            # 3. Tracking & motion estimation
            tracked_objects = tracker.update(detections, frame_idx)

            # Record tracking data
            current_frame_speeds = []
            for obj in tracked_objects:
                current_frame_speeds.append(obj.instantaneous_speed)
                tracking_records.append({
                    "frame_idx": frame_idx,
                    "object_id": obj.object_id,
                    "class_name": obj.class_name,
                    "center_x": obj.center[0],
                    "center_y": obj.center[1],
                    "box_x1": obj.box[0],
                    "box_y1": obj.box[1],
                    "box_x2": obj.box[2],
                    "box_y2": obj.box[3],
                    "displacement_dx": round(obj.dx, 2),
                    "displacement_dy": round(obj.dy, 2),
                    "instantaneous_speed_px_per_frame": round(obj.instantaneous_speed, 2),
                    "cumulative_distance_px": round(obj.cumulative_distance, 2)
                })

            avg_motion_speed = float(np.mean(current_frame_speeds)) if current_frame_speeds else 0.0
            motion_speeds.append(avg_motion_speed)

            # 4. Optical flow analysis
            flow_vis, avg_flow_mag, _ = flow_analyzer.process_frame(frame)
            optical_flow_mags.append(avg_flow_mag)

            # 5. Composite annotated frame
            annotated_frame = draw_annotated_frame(
                frame, tracked_objects, frame_idx, total_frames,
                avg_motion_speed, avg_flow_mag
            )

            if annotated_writer is not None:
                annotated_writer.write(annotated_frame)
            if flow_writer is not None:
                flow_writer.write(flow_vis)

            # Terminal progress reporting
            if processed_count % 15 == 0 or frame_idx == max_frames:
                print(f"Frame: {frame_idx:4d}/{total_frames} | "
                      f"Tracked: {len(tracked_objects):2d} | "
                      f"Motion: {avg_motion_speed:5.2f} px/frame | "
                      f"Optical Flow: {avg_flow_mag:5.2f} px/frame")

    finally:
        cap.release()
        if annotated_writer is not None:
            annotated_writer.release()
        if flow_writer is not None:
            flow_writer.release()

    # Save CSV tracking data
    csv_path = os.path.join(out_dir, "tracking.csv")
    save_tracking_csv(tracking_records, csv_path)

    # Save summary JSON
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "video",
        "input_file": os.path.abspath(input_path),
        "total_video_frames": total_frames,
        "processed_frames": processed_count,
        "fps": round(fps, 2),
        "resolution": {"width": proc_w, "height": proc_h},
        "detector": detector.active_method,
        "optical_flow": args.optical_flow,
        "total_objects_tracked": tracker.next_object_id - 1,
        "average_motion_speed_px_per_frame": round(float(np.mean(motion_speeds)) if motion_speeds else 0.0, 3),
        "average_optical_flow_magnitude": round(float(np.mean(optical_flow_mags)) if optical_flow_mags else 0.0, 3),
        "speed_units": "px/frame (uncalibrated camera)",
        "output_files": {
            "annotated_video": os.path.abspath(annotated_video_path) if args.save_video else None,
            "optical_flow_video": os.path.abspath(flow_video_path) if args.save_video else None,
            "tracking_csv": os.path.abspath(csv_path)
        }
    }
    save_summary_json(summary, os.path.join(out_dir, "summary.json"))

    print(f"\n[VisionTrack] Video processing finished successfully!")
    print(f"[VisionTrack] Total Unique Objects Tracked: {tracker.next_object_id - 1}")
    print(f"[VisionTrack] Tracking Data: {csv_path}")
    print(f"[VisionTrack] Summary Report: {os.path.join(out_dir, 'summary.json')}\n")
    return 0


def run_stereo_pipeline(args: argparse.Namespace) -> int:
    """Executes the stereo depth estimation pipeline."""
    left_path = args.left
    right_path = args.right

    if not left_path or not os.path.isfile(left_path):
        print(f"ERROR: Left stereo image '{left_path}' was not found.")
        return 1
    if not right_path or not os.path.isfile(right_path):
        print(f"ERROR: Right stereo image '{right_path}' was not found.")
        return 1

    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)

    if left_img is None or right_img is None:
        print("ERROR: Failed to decode stereo image pair.")
        return 1

    out_dir = create_experiment_dir(base_output_dir=args.output_dir, mode="stereo")
    print(f"\n=======================================================")
    print(f"VisionTrack: Stereo Depth Estimation")
    print(f"Left Image: {left_path} | Right Image: {right_path}")
    print(f"Matcher: {args.stereo_method.upper()}")
    print(f"Output Directory: {out_dir}")
    print(f"=======================================================\n")

    estimator = StereoDepthEstimator(
        method=args.stereo_method, num_disparities=args.num_disparities,
        block_size=args.stereo_block_size
    )

    norm_disp, depth_vis, metrics = estimator.compute_disparity(left_img, right_img)

    cv2.imwrite(os.path.join(out_dir, "disparity_map.png"), norm_disp)
    cv2.imwrite(os.path.join(out_dir, "depth_map.png"), depth_vis)

    save_summary_json(metrics, os.path.join(out_dir, "summary.json"))
    print(f"[VisionTrack] Disparity and depth map saved to: {out_dir}\n")
    return 0


def generate_samples_command() -> int:
    """Generates synthetic road scene media in data/ for testing."""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    img_path = os.path.join(data_dir, "sample_image.jpg")
    vid_path = os.path.join(data_dir, "sample_traffic.mp4")
    left_path = os.path.join(data_dir, "left.jpg")
    right_path = os.path.join(data_dir, "right.jpg")

    print(f"\n[Generator] Creating sample test road image: {img_path}...")
    generate_synthetic_road_image(img_path, width=640, height=360)

    print(f"[Generator] Creating sample traffic video (120 frames): {vid_path}...")
    generate_synthetic_traffic_video(vid_path, num_frames=120, fps=30, width=640, height=360)

    print(f"[Generator] Creating sample stereo pair: {left_path}, {right_path}...")
    generate_synthetic_stereo_pair(left_path, right_path, width=640, height=360)

    print("\n[Generator] Sample test data generated in data/!")
    print("Ready to run:")
    print("  python main.py --mode image --input data/sample_image.jpg")
    print("  python main.py --mode video --input data/sample_traffic.mp4")
    print("  python main.py --mode stereo --left data/left.jpg --right data/right.jpg\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="VisionTrack",
        description="VisionTrack: Road Scene Analysis and Motion Tracking System",
        epilog="Standalone Computer Vision CLI application for road scene and video analysis."
    )

    # Core Options
    parser.add_argument("--mode", type=str, choices=["image", "video", "stereo", "demo"],
                        default="video", help="Operating mode: image, video, stereo, or demo.")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input image or video file.")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Base output directory for saved results.")

    # Image Pipeline Options
    parser.add_argument("--segment", type=str, choices=["kmeans", "otsu", "adaptive", "region_growing", "meanshift"],
                        default="kmeans", help="Segmentation method to apply in image mode.")
    parser.add_argument("--kmeans-k", type=int, default=4,
                        help="Number of clusters K for K-Means segmentation.")
    parser.add_argument("--canny-low", type=float, default=50.0,
                        help="Lower threshold for Canny edge detector.")
    parser.add_argument("--canny-high", type=float, default=150.0,
                        help="Upper threshold for Canny edge detector.")
    parser.add_argument("--perspective", action="store_true",
                        help="Enable bird's-eye view perspective transformation.")

    # Video Pipeline Options
    parser.add_argument("--detector", type=str, choices=["auto", "yolo", "hog_svm", "motion"],
                        default="auto", help="Detection backend: 'auto', 'yolo', 'hog_svm', or 'motion'.")
    parser.add_argument("--conf-thresh", type=float, default=0.35,
                        help="Detection confidence threshold.")
    parser.add_argument("--bg-method", type=str, choices=["mog2", "knn"],
                        default="mog2", help="Background subtraction algorithm.")
    parser.add_argument("--bg-history", type=int, default=300,
                        help="History length for background modeling.")
    parser.add_argument("--bg-var-threshold", type=float, default=16.0,
                        help="Variance threshold for background model.")
    parser.add_argument("--min-area", type=float, default=350.0,
                        help="Minimum contour area for detected moving objects.")
    parser.add_argument("--max-disappeared", type=int, default=15,
                        help="Maximum frames an object can be lost before track termination.")
    parser.add_argument("--max-distance", type=float, default=85.0,
                        help="Maximum centroid distance in pixels for tracking association.")
    parser.add_argument("--optical-flow", type=str, choices=["sparse", "dense", "none"],
                        default="sparse", help="Optical flow method: 'sparse' (Lucas-Kanade) or 'dense' (Farneback).")

    # Stereo Pipeline Options
    parser.add_argument("--left", type=str, default=None,
                        help="Left image for binocular stereo depth estimation.")
    parser.add_argument("--right", type=str, default=None,
                        help="Right image for binocular stereo depth estimation.")
    parser.add_argument("--stereo-method", type=str, choices=["sgbm", "bm"],
                        default="sgbm", help="Stereo matcher algorithm ('sgbm' or 'bm').")
    parser.add_argument("--num-disparities", type=int, default=64,
                        help="Disparity search range (multiple of 16).")
    parser.add_argument("--stereo-block-size", type=int, default=9,
                        help="Matched block size (odd integer).")

    # Performance & Diagnostics
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="Process every Nth frame (e.g. 2 for 2x faster execution).")
    parser.add_argument("--max-frames", type=int, default=-1,
                        help="Maximum frames to process (-1 for all).")
    parser.add_argument("--resize", type=int, default=None,
                        help="Optional target width to resize frames to.")
    parser.add_argument("--save-video", action="store_true", default=True,
                        help="Save output video files.")
    parser.add_argument("--no-save-video", action="store_false", dest="save_video",
                        help="Do not save output video files.")
    parser.add_argument("--generate-samples", action="store_true",
                        help="Generate small sample test media in data/.")

    return parser


def main() -> int:
    """CLI entrypoint."""
    parser = build_parser()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 0

    args = parser.parse_args()

    if args.generate_samples:
        return generate_samples_command()

    if args.mode == "demo":
        data_image = os.path.join("data", "sample_image.jpg")
        data_traffic = os.path.join("data", "sample_traffic.mp4")
        if not os.path.isfile(data_image) or not os.path.isfile(data_traffic):
            generate_samples_command()

        print("\n[VisionTrack Demo] Running Image Analysis...")
        args.input = data_image
        args.mode = "image"
        args.perspective = True
        run_image_pipeline(args)

        print("\n[VisionTrack Demo] Running Video Analysis...")
        args.input = data_traffic
        args.mode = "video"
        args.max_frames = 60
        return run_video_pipeline(args)

    if args.mode == "image":
        return run_image_pipeline(args)
    elif args.mode == "video":
        return run_video_pipeline(args)
    elif args.mode == "stereo":
        return run_stereo_pipeline(args)
    else:
        print(f"ERROR: Unknown mode '{args.mode}'.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
