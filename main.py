#!/usr/bin/env python3
"""VisionTrack: Computer Vision-Based Road Scene Analysis and Motion Tracking System.

Academic Computer Vision project demonstrating:
- Digital image formation, low-level filtering, geometric transformations (Module 1)
- Frequency-domain Fourier analysis and filtering (Module 1)
- Histogram processing and contrast enhancement (Module 1)
- Epipolar geometry and binocular stereopsis depth estimation (Module 2)
- Projective transformations, homography, bird's-eye view mapping (Module 1 & 2)
- Edge detection, Hough transform lines, Harris corners, HOG, SIFT (Module 3)
- Classical segmentation: Otsu thresholding, K-Means clustering, Region Growing, Mean-Shift (Module 3 & 4)
- Dynamic background modeling (MOG2, KNN) with morphological filtering (Module 4)
- Road-scene object detection (YOLO, HOG+SVM, Motion-blob) (Module 3 & 4)
- Multi-object tracking with persistent IDs and motion parameter estimation (Module 4)
- Sparse Lucas-Kanade and dense Farneback optical flow (Module 4)

Usage examples:
    python main.py --mode image --input data/sample_road.jpg
    python main.py --mode image --input data/sample_road.jpg --features harris --segment kmeans
    python main.py --mode video --input data/sample_traffic.mp4
    python main.py --mode stereo --left data/left_stereo.jpg --right data/right_stereo.jpg
    python main.py --generate-samples
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Strictly headless backend
import matplotlib.pyplot as plt

# Internal VisionTrack modules
from src.preprocessing import (
    to_grayscale, apply_gaussian_filter, apply_median_filter,
    apply_bilateral_filter, resize_image
)
from src.enhancement import (
    equalize_histogram_gray, equalize_histogram_color,
    apply_clahe, save_histogram_plot, compute_histogram_metrics
)
from src.frequency import (
    compute_fft, apply_frequency_filter, save_fourier_analysis_plot
)
from src.features import (
    detect_canny_edges, detect_laplacian_of_gaussian,
    detect_hough_lines, detect_harris_corners,
    extract_hog_features, extract_sift_features
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
    """Executes the complete classical computer vision image analysis pipeline."""
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
    exp_dir = create_experiment_dir(base_output_dir=args.output_dir, mode="image")
    print(f"\n=======================================================")
    print(f"VisionTrack: Image Scene Analysis Pipeline")
    print(f"Input: {input_path} ({w}x{h})")
    print(f"Output Directory: {exp_dir}")
    print(f"=======================================================\n")

    summary: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "image",
        "input_file": os.path.abspath(input_path),
        "resolution": {"width": w, "height": h},
        "selected_features": args.features,
        "selected_segmentation": args.segment
    }

    # 1. Original Image
    cv2.imwrite(os.path.join(exp_dir, "01_original.jpg"), img_bgr)

    # 2. Grayscale conversion (Module 1)
    print("[1/7] Performing grayscale conversion (ITU-R BT.601)...")
    gray = to_grayscale(img_bgr)
    cv2.imwrite(os.path.join(exp_dir, "02_grayscale.jpg"), gray)

    # 3. Spatial Filtering: Gaussian & Median (Module 1)
    print("[2/7] Applying low-level convolution filters (Gaussian, Median, Bilateral)...")
    gaussian = apply_gaussian_filter(gray, kernel_size=5, sigma=1.2)
    median = apply_median_filter(gray, kernel_size=5)
    bilateral = apply_bilateral_filter(img_bgr, diameter=9, sigma_color=75, sigma_space=75)
    cv2.imwrite(os.path.join(exp_dir, "03_filtered_gaussian.jpg"), gaussian)
    cv2.imwrite(os.path.join(exp_dir, "04_filtered_median.jpg"), median)
    cv2.imwrite(os.path.join(exp_dir, "05_filtered_bilateral.jpg"), bilateral)

    # 4. Enhancement & Histogram Processing (Module 1)
    print("[3/7] Performing contrast enhancement & histogram equalization...")
    equalized_gray = equalize_histogram_gray(gray)
    equalized_color = equalize_histogram_color(img_bgr)
    clahe_gray = apply_clahe(gray, clip_limit=2.5)
    cv2.imwrite(os.path.join(exp_dir, "06_enhanced_hist_equalized.jpg"), equalized_gray)
    cv2.imwrite(os.path.join(exp_dir, "07_enhanced_color_equalized.jpg"), equalized_color)
    cv2.imwrite(os.path.join(exp_dir, "08_enhanced_clahe.jpg"), clahe_gray)

    hist_metrics = compute_histogram_metrics(gray)
    summary["histogram_metrics"] = hist_metrics
    save_histogram_plot(gray, equalized_gray, os.path.join(exp_dir, "09_histogram_distribution.png"))

    # 5. Frequency Domain Analysis (Module 1 Fourier Transform)
    print("[4/7] Computing 2D Discrete Fourier Transform & frequency filtering...")
    save_fourier_analysis_plot(gray, os.path.join(exp_dir, "10_fourier_analysis.png"), cutoff_d0=35.0)

    # 6. Feature Extraction (Module 3)
    print(f"[5/7] Extracting computer vision features (Requested: {args.features})...")
    edges_canny = detect_canny_edges(gaussian, low_threshold=args.canny_low, high_threshold=args.canny_high)
    cv2.imwrite(os.path.join(exp_dir, "11_features_canny_edges.jpg"), edges_canny)

    edges_log = detect_laplacian_of_gaussian(gray, ksize=5, sigma=1.0)
    cv2.imwrite(os.path.join(exp_dir, "12_features_log_edges.jpg"), edges_log)

    hough_vis, lines = detect_hough_lines(gray, canny_low=args.canny_low, canny_high=args.canny_high)
    cv2.imwrite(os.path.join(exp_dir, "13_features_hough_lines.jpg"), hough_vis)
    summary["hough_lines_detected"] = len(lines)

    harris_vis, _, num_corners = detect_harris_corners(gray)
    cv2.imwrite(os.path.join(exp_dir, "14_features_harris_corners.jpg"), harris_vis)
    summary["harris_corners_detected"] = num_corners

    hog_vec, hog_vis = extract_hog_features(gray)
    cv2.imwrite(os.path.join(exp_dir, "15_features_hog_gradient_field.jpg"), hog_vis)
    summary["hog_vector_dimension"] = len(hog_vec)

    sift_vis, sift_kps, _ = extract_sift_features(gray)
    cv2.imwrite(os.path.join(exp_dir, "16_features_sift_keypoints.jpg"), sift_vis)
    summary["sift_keypoints_detected"] = len(sift_kps)

    # 7. Image Segmentation (Module 3 & 4)
    print(f"[6/7] Executing segmentation algorithms (Selected: {args.segment})...")
    seg_otsu, otsu_val = segment_by_threshold(gray, method="otsu")
    seg_adaptive, _ = segment_by_threshold(gray, method="adaptive")
    seg_kmeans, _ = segment_by_kmeans(img_bgr, k=args.kmeans_k)
    seg_region = segment_by_region_growing(gray)
    seg_meanshift = segment_by_mean_shift(img_bgr)

    cv2.imwrite(os.path.join(exp_dir, "17_segmentation_otsu.jpg"), seg_otsu)
    cv2.imwrite(os.path.join(exp_dir, "18_segmentation_adaptive.jpg"), seg_adaptive)
    cv2.imwrite(os.path.join(exp_dir, "19_segmentation_kmeans.jpg"), seg_kmeans)
    cv2.imwrite(os.path.join(exp_dir, "20_segmentation_region_growing.jpg"), seg_region)
    cv2.imwrite(os.path.join(exp_dir, "21_segmentation_meanshift.jpg"), seg_meanshift)

    summary["segmentation_otsu_threshold"] = otsu_val

    # Comparison figure for all 4 segmentation methods
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(seg_otsu, cmap="gray")
    axes[0, 0].set_title(f"Otsu Threshold (T={otsu_val:.1f})")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(seg_kmeans, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"K-Means Clustering (K={args.kmeans_k})")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(seg_region, cmap="gray")
    axes[1, 0].set_title("Seeded Region Growing")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cv2.cvtColor(seg_meanshift, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Mean Shift Segmentation")
    axes[1, 1].axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(exp_dir, "22_segmentation_comparison.png"), dpi=150)
    plt.close(fig)

    # 8. Perspective Transformation / Bird's-Eye View (Optional / Module 1 & 2)
    if args.perspective:
        print("[7/7] Computing Bird's-Eye View perspective transformation...")
        pt = PerspectiveTransformer()
        bev_img = pt.warp_to_birds_eye(img_bgr)
        roi_vis = pt.draw_roi_on_image(img_bgr)
        cv2.imwrite(os.path.join(exp_dir, "23_perspective_road_roi.jpg"), roi_vis)
        cv2.imwrite(os.path.join(exp_dir, "24_perspective_birds_eye_view.jpg"), bev_img)
        summary["perspective_transform_applied"] = True

    # Save summary JSON
    save_summary_json(summary, os.path.join(exp_dir, "summary.json"))
    print(f"\n[VisionTrack] Image analysis pipeline completed successfully!")
    print(f"[VisionTrack] All visual artifacts and summary.json saved to: {exp_dir}\n")
    return 0


def run_video_pipeline(args: argparse.Namespace) -> int:
    """Executes the complete video analysis, motion tracking, and optical flow pipeline."""
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

    exp_dir = create_experiment_dir(base_output_dir=args.output_dir, mode="video")
    sample_frames_dir = os.path.join(exp_dir, "sample_frames")
    os.makedirs(sample_frames_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"VisionTrack: Video Scene Analysis & Tracking Pipeline")
    print(f"Input Video: {input_path}")
    print(f"Total Frames: {total_frames} | FPS: {fps:.2f} | Resolution: {proc_w}x{proc_h}")
    print(f"Detector Engine: {args.detector} | Optical Flow: {args.optical_flow}")
    print(f"Output Directory: {exp_dir}")
    print(f"=======================================================\n")

    # Initialize processing sub-systems
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
    annotated_writer = None
    flow_writer = None
    mask_writer = None

    annotated_video_path = os.path.join(exp_dir, "annotated_video.mp4")
    flow_video_path = os.path.join(exp_dir, "optical_flow.mp4")
    mask_video_path = os.path.join(exp_dir, "foreground_mask.mp4")

    if args.save_video:
        annotated_writer = create_video_writer(annotated_video_path, fps, (proc_w, proc_h))
        flow_writer = create_video_writer(flow_video_path, fps, (proc_w, proc_h))
        mask_writer = create_video_writer(mask_video_path, fps, (proc_w, proc_h))

    tracking_records: List[Dict[str, Any]] = []
    frame_idx = 0
    processed_count = 0
    motion_speeds: List[float] = []
    optical_flow_mags: List[float] = []
    max_frames_to_process = args.max_frames if args.max_frames > 0 else total_frames

    print("Processing video frames...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if args.max_frames > 0 and frame_idx > args.max_frames:
                break

            # Frame skipping for computational throughput
            if frame_idx % args.frame_skip != 0:
                continue

            processed_count += 1
            if args.resize:
                frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

            # 1. Background Subtraction & Morphological Filtering (Module 4)
            raw_mask, cleaned_mask = bg_subtractor.apply(frame)
            moving_regions = bg_subtractor.extract_moving_regions(cleaned_mask)

            # 2. Object Detection (Module 3 & 4)
            detections = detector.detect(frame, motion_regions=moving_regions)

            # 3. Multi-Object Tracking & Motion Parameter Estimation (Module 4)
            tracked_objects = tracker.update(detections, frame_idx)

            # Record tracking parameters
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

            # 4. Optical Flow Analysis (Module 4)
            flow_vis, avg_flow_mag, _ = flow_analyzer.process_frame(frame)
            optical_flow_mags.append(avg_flow_mag)

            # 5. Composite Visualizations & Headless Writing
            annotated_frame = draw_annotated_frame(
                frame, tracked_objects, frame_idx, total_frames,
                avg_motion_speed, avg_flow_mag
            )

            # Mask 3-channel visualization
            cleaned_mask_bgr = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)

            if annotated_writer is not None:
                annotated_writer.write(annotated_frame)
            if flow_writer is not None:
                flow_writer.write(flow_vis)
            if mask_writer is not None:
                mask_writer.write(cleaned_mask_bgr)

            # Periodically save keyframe snapshots (e.g. every 25 processed frames)
            if processed_count % 25 == 1 or frame_idx == max_frames_to_process:
                cv2.imwrite(os.path.join(sample_frames_dir, f"frame_{frame_idx:05d}_annotated.jpg"), annotated_frame)
                cv2.imwrite(os.path.join(sample_frames_dir, f"frame_{frame_idx:05d}_mask.jpg"), cleaned_mask)
                cv2.imwrite(os.path.join(sample_frames_dir, f"frame_{frame_idx:05d}_optical_flow.jpg"), flow_vis)

            # Console Progress reporting
            if processed_count % 15 == 0 or frame_idx == max_frames_to_process:
                print(f"Frame: {frame_idx:4d}/{total_frames} | "
                      f"Detected: {len(detections):2d} | "
                      f"Tracked: {len(tracked_objects):2d} | "
                      f"Mean Motion: {avg_motion_speed:5.2f} px/f | "
                      f"Flow Mag: {avg_flow_mag:5.2f}")

    finally:
        cap.release()
        if annotated_writer is not None:
            annotated_writer.release()
        if flow_writer is not None:
            flow_writer.release()
        if mask_writer is not None:
            mask_writer.release()

    # Save CSV tracking data
    csv_path = os.path.join(exp_dir, "tracking.csv")
    save_tracking_csv(tracking_records, csv_path)

    # Save JSON summary report
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "video",
        "input_file": os.path.abspath(input_path),
        "total_video_frames": total_frames,
        "processed_frames": processed_count,
        "video_fps": round(fps, 2),
        "resolution": {"width": proc_w, "height": proc_h},
        "detector_engine": detector.active_method,
        "optical_flow_method": args.optical_flow,
        "total_unique_objects_tracked": tracker.next_object_id - 1,
        "overall_average_pixel_motion_speed": round(float(np.mean(motion_speeds)) if motion_speeds else 0.0, 3),
        "overall_average_optical_flow_magnitude": round(float(np.mean(optical_flow_mags)) if optical_flow_mags else 0.0, 3),
        "motion_unit_disclaimer": "Pixel motion and approximate speed are reported in pixel/frame units. Metric speed (km/h) requires extrinsic camera calibration.",
        "output_files": {
            "annotated_video": os.path.abspath(annotated_video_path) if args.save_video else None,
            "optical_flow_video": os.path.abspath(flow_video_path) if args.save_video else None,
            "mask_video": os.path.abspath(mask_video_path) if args.save_video else None,
            "tracking_csv": os.path.abspath(csv_path),
            "sample_frames_dir": os.path.abspath(sample_frames_dir)
        }
    }
    save_summary_json(summary, os.path.join(exp_dir, "summary.json"))

    print(f"\n[VisionTrack] Video processing finished successfully!")
    print(f"[VisionTrack] Total Unique Objects Tracked: {tracker.next_object_id - 1}")
    print(f"[VisionTrack] Tracking Data: {csv_path}")
    print(f"[VisionTrack] Experiment Summary: {os.path.join(exp_dir, 'summary.json')}\n")
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

    exp_dir = create_experiment_dir(base_output_dir=args.output_dir, mode="stereo")
    print(f"\n=======================================================")
    print(f"VisionTrack: Stereo Depth Estimation (Module 2)")
    print(f"Left Image: {left_path} | Right Image: {right_path}")
    print(f"Matcher Method: {args.stereo_method}")
    print(f"Output Directory: {exp_dir}")
    print(f"=======================================================\n")

    estimator = StereoDepthEstimator(
        method=args.stereo_method, num_disparities=args.num_disparities,
        block_size=args.stereo_block_size
    )

    norm_disp, depth_vis, metrics = estimator.compute_disparity(left_img, right_img)

    cv2.imwrite(os.path.join(exp_dir, "01_left_rectified.jpg"), left_img)
    cv2.imwrite(os.path.join(exp_dir, "02_right_rectified.jpg"), right_img)
    cv2.imwrite(os.path.join(exp_dir, "03_disparity_map.png"), norm_disp)
    cv2.imwrite(os.path.join(exp_dir, "04_depth_visualization.png"), depth_vis)

    # Save side-by-side comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Left Rectified View")
    axes[0].axis("off")

    axes[1].imshow(norm_disp, cmap="gray")
    axes[1].set_title(f"Disparity Map ({args.stereo_method.upper()})")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Relative Depth Map (Inferno)")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(exp_dir, "05_stereo_comparison.png"), dpi=150)
    plt.close(fig)

    save_summary_json(metrics, os.path.join(exp_dir, "summary.json"))
    print(f"[VisionTrack] Disparity and depth map saved to: {exp_dir}\n")
    return 0


def generate_samples_command() -> int:
    """Generates synthetic road scene images, videos, and stereo pairs in data/."""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    img_path = os.path.join(data_dir, "sample_road.jpg")
    vid_path = os.path.join(data_dir, "sample_traffic.mp4")
    left_path = os.path.join(data_dir, "left_stereo.jpg")
    right_path = os.path.join(data_dir, "right_stereo.jpg")

    print(f"\n[Generator] Creating synthetic test road scene image: {img_path}...")
    generate_synthetic_road_image(img_path, width=640, height=360)

    print(f"[Generator] Creating synthetic traffic video (120 frames): {vid_path}...")
    generate_synthetic_traffic_video(vid_path, num_frames=120, fps=30, width=640, height=360)

    print(f"[Generator] Creating synthetic binocular stereo pair: {left_path}, {right_path}...")
    generate_synthetic_stereo_pair(left_path, right_path, width=640, height=360)

    print("\n[Generator] All synthetic test data successfully generated in data/!")
    print("Ready to run:")
    print("  python main.py --mode image --input data/sample_road.jpg")
    print("  python main.py --mode video --input data/sample_traffic.mp4")
    print("  python main.py --mode stereo --left data/left_stereo.jpg --right data/right_stereo.jpg\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Builds the comprehensive CLI parser."""
    parser = argparse.ArgumentParser(
        prog="VisionTrack",
        description="VisionTrack: Computer Vision-Based Road Scene Analysis and Motion Tracking System",
        epilog="Academic Computer Vision Project covering Modules 1-4 of the Computer Vision Syllabus."
    )

    # Core Execution Mode
    parser.add_argument("--mode", type=str, choices=["image", "video", "stereo", "demo"],
                        default="video", help="Operating mode: image, video, stereo, or demo.")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input image or video file.")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Base output directory where timestamped experiment runs are saved.")

    # Feature Extraction & Segmentation (Module 3 & 4)
    parser.add_argument("--features", type=str, choices=["canny", "harris", "hog", "sift", "all"],
                        default="all", help="Feature extraction method for image mode.")
    parser.add_argument("--segment", type=str, choices=["otsu", "adaptive", "kmeans", "region_growing", "meanshift"],
                        default="kmeans", help="Segmentation method to showcase in image mode.")
    parser.add_argument("--kmeans-k", type=int, default=4,
                        help="Number of color clusters K for K-Means segmentation.")
    parser.add_argument("--canny-low", type=float, default=50.0,
                        help="Lower hysteresis threshold for Canny edge detection.")
    parser.add_argument("--canny-high", type=float, default=150.0,
                        help="Upper hysteresis threshold for Canny edge detection.")

    # Video Pipeline: Detection, Background, Tracking, Flow (Module 4)
    parser.add_argument("--detector", type=str, choices=["auto", "yolo", "hog_svm", "motion"],
                        default="auto", help="Object detection backend (auto, yolo, hog_svm, or classical motion).")
    parser.add_argument("--conf-thresh", type=float, default=0.35,
                        help="Confidence threshold for object detection.")
    parser.add_argument("--bg-method", type=str, choices=["mog2", "knn"],
                        default="mog2", help="Background subtraction method (MOG2 or KNN).")
    parser.add_argument("--bg-history", type=int, default=300,
                        help="Number of frames observed by background subtractor.")
    parser.add_argument("--bg-var-threshold", type=float, default=16.0,
                        help="Variance threshold for background pixel classification.")
    parser.add_argument("--min-area", type=float, default=350.0,
                        help="Minimum contour pixel area for valid moving objects.")
    parser.add_argument("--max-disappeared", type=int, default=15,
                        help="Maximum frames an object can be lost before deregistering.")
    parser.add_argument("--max-distance", type=float, default=85.0,
                        help="Maximum centroid distance in pixels for tracking association.")
    parser.add_argument("--optical-flow", type=str, choices=["sparse", "dense", "none"],
                        default="sparse", help="Optical flow motion analysis method (sparse LK or dense Farneback).")

    # Geometry & Perspective (Module 1 & 2)
    parser.add_argument("--perspective", action="store_true",
                        help="Enable Bird's-Eye View perspective transformation in image mode.")

    # Stereo Depth (Module 2)
    parser.add_argument("--left", type=str, default=None,
                        help="Left image for binocular stereo depth estimation.")
    parser.add_argument("--right", type=str, default=None,
                        help="Right image for binocular stereo depth estimation.")
    parser.add_argument("--stereo-method", type=str, choices=["sgbm", "bm"],
                        default="sgbm", help="Stereo matching algorithm (SGBM or BM).")
    parser.add_argument("--num-disparities", type=int, default=64,
                        help="Disparity search range for stereo matching (multiple of 16).")
    parser.add_argument("--stereo-block-size", type=int, default=9,
                        help="Window block size for stereo correlation matching (odd integer).")

    # Performance & Diagnostics
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="Process every Nth frame (e.g. 2 for 2x faster CPU throughput).")
    parser.add_argument("--max-frames", type=int, default=-1,
                        help="Maximum number of frames to process (-1 for all).")
    parser.add_argument("--resize", type=int, default=None,
                        help="Optional frame width to resize input to (preserves aspect ratio).")
    parser.add_argument("--save-video", action="store_true", default=True,
                        help="Save processed output videos to disk.")
    parser.add_argument("--no-save-video", action="store_false", dest="save_video",
                        help="Do not write output video files (faster processing).")
    parser.add_argument("--generate-samples", action="store_true",
                        help="Generate synthetic road image, traffic video, and stereo pairs in data/.")

    return parser


def main() -> int:
    """CLI entrypoint."""
    parser = build_parser()

    # Handle --help or empty args gracefully
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 0

    args = parser.parse_args()

    if args.generate_samples:
        return generate_samples_command()

    if args.mode == "demo":
        # Generate samples if missing, then run image and video demos
        data_road = os.path.join("data", "sample_road.jpg")
        data_traffic = os.path.join("data", "sample_traffic.mp4")
        if not os.path.isfile(data_road) or not os.path.isfile(data_traffic):
            generate_samples_command()

        print("\n[VisionTrack Demo] Running Image Pipeline Demo...")
        args.input = data_road
        args.mode = "image"
        args.perspective = True
        run_image_pipeline(args)

        print("\n[VisionTrack Demo] Running Video Pipeline Demo...")
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
