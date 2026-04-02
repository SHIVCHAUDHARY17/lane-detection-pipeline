import cv2
import numpy as np


def apply_canny(mask: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Apply Gaussian blur followed by Canny edge detection.

    We blur first to remove noise — without this every tiny pixel
    variation triggers a false edge. Canny then finds strong brightness
    gradients which correspond to real structural boundaries.

    Args:
        mask: binary mask from color filtering (output of get_lane_mask)
        cfg: canny section from config YAML

    Returns:
        Binary edge image — white pixels are detected edges
    """
    # Step 1: blur to suppress noise
    # kernel size must be odd — controls how much smoothing happens
    blurred = cv2.GaussianBlur(
        mask,
        (cfg["blur_kernel"], cfg["blur_kernel"]),
        0
    )

    # Step 2: Canny edge detection
    # low_threshold: below this = not an edge
    # high_threshold: above this = definitely an edge
    # between the two = edge only if connected to a strong edge
    edges = cv2.Canny(blurred, cfg["low_threshold"], cfg["high_threshold"])

    return edges


def get_roi_vertices(height: int, width: int, cfg: dict) -> np.ndarray:
    """
    Calculate ROI triangle vertices from config fractions.

    Instead of hardcoding pixel coordinates (which break if resolution
    changes), we store positions as fractions of frame size in config.
    This makes the ROI resolution-independent.

    The triangle points are:
        - bottom-left corner of frame
        - bottom-right corner of frame
        - apex at centre-top (where the road vanishes)

    Args:
        height: frame height in pixels
        width: frame width in pixels
        cfg: roi section from config YAML

    Returns:
        Array of vertices shape (1, 3, 2) — required by fillPoly
    """
    bottom_left  = [0, height]
    bottom_right = [width, height]
    apex         = [width // 2, int(height * cfg["top_left_y"])]

    return np.array([[bottom_left, bottom_right, apex]], dtype=np.int32)


def apply_roi(edges: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """
    Mask the edge image to only keep edges inside the ROI polygon.

    Steps:
    1. Create a completely black image same size as edges
    2. Fill the ROI polygon with white
    3. AND with edges — only edges inside polygon survive

    Args:
        edges: Canny edge output
        vertices: ROI polygon vertices from get_roi_vertices

    Returns:
        Edge image with everything outside ROI removed
    """
    # Black canvas same size as edges
    mask = np.zeros_like(edges)

    # Fill our triangle with white (255)
    cv2.fillPoly(mask, vertices, 255)

    # Keep only edges that fall inside the white triangle
    roi_edges = cv2.bitwise_and(edges, mask)

    return roi_edges