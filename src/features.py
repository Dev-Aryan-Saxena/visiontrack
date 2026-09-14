"""Feature extraction utilities for edges, lines, corners, and descriptors.

Implements classical feature extraction techniques including gradient-based edge
detection (Canny, LoG, DoG), parametric lane and line detection (Hough Transform),
second-moment corner detection (Harris), dense gradient descriptors (HOG), and
scale-space invariant keypoints (SIFT).
"""

from typing import Tuple, List, Dict, Any, Optional
import cv2
import numpy as np


def detect_canny_edges(gray_image: np.ndarray, low_threshold: float = 50.0,
                       high_threshold: float = 150.0, aperture_size: int = 3,
                       l2_gradient: bool = True) -> np.ndarray:
    """Detects edges using the 4-stage optimal Canny edge detection algorithm.

    Algorithm Explanation:
    1. What it does: Produces a thin, continuous binary edge map with minimal false responses.
    2. Why it is used: In road scenes, delineates road boundaries, lane markings, vehicle silhouettes,
       and pedestrian contours.
    3. Mathematical Basis (John F. Canny, 1986):
       - Stage 1: Gaussian smoothing to suppress high-frequency noise.
       - Stage 2: Gradient magnitude M = sqrt(Gx^2 + Gy^2) and orientation theta = arctan2(Gy, Gx) via Sobel.
       - Stage 3: Non-Maximum Suppression (NMS) along the gradient direction to thin edges to 1-pixel width.
       - Stage 4: Hysteresis Thresholding with two thresholds (T_low, T_high). Pixels > T_high are strong edges;
         pixels between T_low and T_high are preserved only if connected to a strong edge via 8-connectivity.
    4. Input: Grayscale image (H x W, uint8), thresholds.
    5. Output: Binary edge map (H x W, uint8, values 0 or 255).
    """
    if len(gray_image.shape) != 2:
        raise ValueError("Input must be a single-channel grayscale image.")
    return cv2.Canny(gray_image, threshold1=low_threshold, threshold2=high_threshold,
                     apertureSize=aperture_size, L2gradient=l2_gradient)


def detect_laplacian_of_gaussian(gray_image: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Detects edges using the Laplacian of Gaussian (LoG) Marr-Hildreth operator.

    Algorithm Explanation:
    1. What it does: Computes the second spatial derivative of the Gaussian-smoothed image and locates zero-crossings.
    2. Why it is used: Localizes rapid intensity transitions via second spatial derivative zero-crossings.
    3. Mathematical Basis: nabla^2 G(x, y) = ((x^2 + y^2 - 2*sigma^2) / sigma^4) * G(x, y).
    """
    if len(gray_image.shape) != 2:
        raise ValueError("Input must be a single-channel grayscale image.")
    smoothed = cv2.GaussianBlur(gray_image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    laplacian = cv2.Laplacian(smoothed, cv2.CV_64F, ksize=ksize)
    abs_laplacian = np.uint8(np.absolute(laplacian))
    return cv2.normalize(abs_laplacian, None, 0, 255, cv2.NORM_MINMAX)


def detect_difference_of_gaussians(gray_image: np.ndarray, sigma1: float = 1.0,
                                   sigma2: float = 1.6) -> np.ndarray:
    """Detects multi-scale bandpass features via Difference of Gaussians (DoG).

    Algorithm Explanation:
    1. What it does: Subtracts two Gaussian-blurred versions of the image with different scales.
    2. Why it is used: Efficient approximation to Laplacian of Gaussian (LoG), forming the foundation
       for scale-space extrema in SIFT.
    3. Mathematical Basis: DoG(x, y, sigma) = G(x, y, k*sigma) - G(x, y, sigma) approx (k - 1) * sigma^2 * nabla^2 G.
    """
    if len(gray_image.shape) != 2:
        raise ValueError("Input must be a single-channel grayscale image.")
    g1 = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=sigma1, sigmaY=sigma1)
    g2 = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=sigma2, sigmaY=sigma2)
    dog = cv2.absdiff(g1, g2)
    return cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)


def detect_hough_lines(gray_image: np.ndarray, canny_low: float = 50.0, canny_high: float = 150.0,
                       rho: float = 1.0, theta_deg: float = 1.0, threshold: int = 50,
                       min_line_length: float = 40.0, max_line_gap: float = 10.0,
                       roi_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """Detects linear features (such as lane markings) using the Probabilistic Hough Transform.

    Algorithm Explanation:
    1. What it does: Maps edge points from Cartesian space (x, y) to Hough parameter accumulator space (rho, theta).
    2. Why it is used: In road scene analysis, straight lane boundaries, highway dividers, and road curbs
       naturally form collinear edge segments.
    3. Mathematical Basis: Normal parameterization rho = x * cos(theta) + y * sin(theta).
       Each edge pixel votes for a sinusoidal curve in the accumulator. Peak intersections correspond to collinear lines.
       The Probabilistic Hough Transform (Matas et al.) randomly samples edge points to achieve O(N) efficiency.
    4. Input: Grayscale image, edge thresholds, line length constraints, optional ROI mask.
    5. Output: Tuple of (visualization BGR image, list of detected line segments [(x1, y1, x2, y2)]).
    """
    edges = detect_canny_edges(gray_image, low_threshold=canny_low, high_threshold=canny_high)
    if roi_mask is not None:
        edges = cv2.bitwise_and(edges, roi_mask)

    theta = np.pi / 180.0 * theta_deg
    lines = cv2.HoughLinesP(edges, rho=rho, theta=theta, threshold=threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    vis = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    detected_lines = []

    if lines is not None:
        for line in lines:
            coords = line.ravel()
            if len(coords) >= 4:
                x1, y1, x2, y2 = coords[:4]
                detected_lines.append((int(x1), int(y1), int(x2), int(y2)))
                cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(vis, (int(x1), int(y1)), 3, (0, 0, 255), -1)
                cv2.circle(vis, (int(x2), int(y2)), 3, (0, 0, 255), -1)

    return vis, detected_lines


def detect_harris_corners(gray_image: np.ndarray, block_size: int = 3, ksize: int = 3,
                          k: float = 0.04, threshold_ratio: float = 0.01) -> Tuple[np.ndarray, np.ndarray, int]:
    """Detects corner interest points using the Harris Corner Detector.

    Algorithm Explanation:
    1. What it does: Identifies 2D interest points where local intensity changes significantly in all directions.
    2. Why it is used: Corners are stable, rotation-invariant features ideal for tracking vehicle corners,
       road signs, and lane vertices across consecutive video frames.
    3. Mathematical Basis (Chris Harris & Mike Stephens, 1988):
       Autocorrelation matrix M = sum_{(x,y) in W} w(x,y) [ Ix^2   Ix*Iy ]
                                                          [ Ix*Iy Iy^2  ]
       Eigenvalues lambda1, lambda2 describe curvature along principal axes.
       Harris Response Function: R = det(M) - k * (trace(M))^2 = lambda1*lambda2 - k*(lambda1 + lambda2)^2.
       - R > threshold: Corner region (both eigenvalues large).
       - R < -threshold: Edge region (one large eigenvalue).
       - |R| small: Flat homogeneous region.
    4. Input: Grayscale image (uint8), block_size, Sobel ksize, sensitivity k (typically 0.04-0.06).
    5. Output: Tuple of (visualization BGR image, response matrix R, number of corners detected).
    """
    if len(gray_image.shape) != 2:
        raise ValueError("Input must be a single-channel grayscale image.")

    gray_float = np.float32(gray_image)
    # Compute Harris response map R
    dst = cv2.cornerHarris(gray_float, blockSize=block_size, ksize=ksize, k=k)

    # Dilate response to locate local maxima (Non-Maximum Suppression)
    dst_dilated = cv2.dilate(dst, None)

    # Threshold condition: must be above threshold_ratio * max(R) and match local maximum
    thresh = threshold_ratio * dst.max() if dst.max() > 0 else 0
    corner_mask = (dst > thresh) & (dst == dst_dilated)

    vis = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    y_coords, x_coords = np.where(corner_mask)
    num_corners = len(x_coords)

    for x, y in zip(x_coords, y_coords):
        cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, (int(x), int(y)), 5, (255, 255, 255), 1, cv2.LINE_AA)

    return vis, dst, num_corners


def extract_hog_features(gray_image: np.ndarray, cell_size: Tuple[int, int] = (8, 8),
                         cells_per_block: Tuple[int, int] = (2, 2),
                         nbins: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """Computes Histogram of Oriented Gradients (HOG) features and renders a visual gradient field.

    Algorithm Explanation:
    1. What it does: Captures local object appearance and shape distributions through local gradient orientations.
    2. Why it is used: Foundational descriptor (Dalal & Triggs, 2005) for human pedestrian and vehicle detection.
    3. Mathematical Basis:
       - Computes 1D gradients Gx, Gy via [-1, 0, 1] filters.
       - Divides image into small spatial cells (e.g. 8x8 pixels).
       - Accumulates 1D histogram of gradient orientations weighted by gradient magnitude into 9 unsigned bins [0, 180).
       - Groups cells into overlapping blocks (e.g. 2x2 cells) and normalizes using L2-Hys norm for illumination invariance.
    4. Output: Tuple of (1D HOG feature vector, visualization BGR image of cell gradient vectors).
    """
    h, w = gray_image.shape[:2]
    # Resize to multiple of cell size for clean block partitioning
    pad_h = (cell_size[0] - (h % cell_size[0])) % cell_size[0]
    pad_w = (cell_size[1] - (w % cell_size[1])) % cell_size[1]
    if pad_h > 0 or pad_w > 0:
        gray_image = cv2.copyMakeBorder(gray_image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    h, w = gray_image.shape[:2]

    # Compute gradients
    gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = ang % 180.0  # Unsigned orientations [0, 180)

    cell_h, cell_w = cell_size
    n_cells_y = h // cell_h
    n_cells_x = w // cell_w

    bin_width = 180.0 / nbins
    cell_histograms = np.zeros((n_cells_y, n_cells_x, nbins), dtype=np.float32)

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            c_mag = mag[cy * cell_h:(cy + 1) * cell_h, cx * cell_w:(cx + 1) * cell_w]
            c_ang = ang[cy * cell_h:(cy + 1) * cell_h, cx * cell_w:(cx + 1) * cell_w]

            bin_idx = (c_ang / bin_width).astype(int) % nbins
            for b in range(nbins):
                cell_histograms[cy, cx, b] = np.sum(c_mag[bin_idx == b])

    # Block normalization (L2-norm)
    block_h, block_w = cells_per_block
    n_blocks_y = n_cells_y - block_h + 1
    n_blocks_x = n_cells_x - block_w + 1
    hog_vector = []

    if n_blocks_y > 0 and n_blocks_x > 0:
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                block = cell_histograms[by:by + block_h, bx:bx + block_w].flatten()
                norm = np.sqrt(np.sum(block ** 2) + 1e-6)
                hog_vector.extend(block / norm)
    else:
        hog_vector = cell_histograms.flatten()

    # Create visualization of gradient cell vectors
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    radius = min(cell_h, cell_w) // 2
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            center_x = cx * cell_w + cell_w // 2
            center_y = cy * cell_h + cell_h // 2
            hist = cell_histograms[cy, cx]
            max_mag = np.max(hist) + 1e-6

            for b in range(nbins):
                strength = hist[b] / max_mag
                if strength > 0.1:
                    angle_rad = (b * bin_width + bin_width / 2.0) * np.pi / 180.0
                    dx = int(radius * strength * np.cos(angle_rad))
                    dy = int(radius * strength * np.sin(angle_rad))
                    color_intensity = int(min(255, 255 * strength))
                    cv2.line(vis, (center_x - dx, center_y - dy),
                             (center_x + dx, center_y + dy),
                             (0, color_intensity, color_intensity), 1, cv2.LINE_AA)

    return np.array(hog_vector, dtype=np.float32), vis


def extract_sift_features(gray_image: np.ndarray, max_keypoints: int = 500) -> Tuple[np.ndarray, Any, Any]:
    """Extracts Scale-Invariant Feature Transform (SIFT) keypoints and descriptors.

    Algorithm Explanation:
    1. What it does: Computes scale-, rotation-, and illumination-invariant local feature descriptors (David Lowe, 2004).
    2. Why it is used: In road scene tracking, matches landmarks, road signs, and objects despite vehicle zoom or turns.
    3. Mathematical Basis:
       - Scale-space extrema detection across octaves of Difference of Gaussians (DoG).
       - Taylor series sub-pixel keypoint refinement and low-contrast/edge response rejection via Hessian.
       - Canonical orientation assignment based on local gradient histograms.
       - 128-dimensional descriptor: 4x4 spatial grid of 8-bin gradient orientation histograms.
    4. Output: Tuple of (visualization image with keypoint orientation/scale circles, keypoints list, descriptors).
    """
    if not hasattr(cv2, "SIFT_create"):
        # SIFT fallback if unavailable
        vis = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        cv2.putText(vis, "SIFT not supported in current OpenCV build", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return vis, [], None

    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    vis = cv2.drawKeypoints(gray_image, keypoints, None,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                            color=(0, 215, 255))
    return vis, keypoints, descriptors
