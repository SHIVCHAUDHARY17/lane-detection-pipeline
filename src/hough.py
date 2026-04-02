import cv2
import numpy as np


def detect_hough_lines(roi_edges: np.ndarray, cfg: dict):
    """
    Detect line segments in the ROI edge image using Hough transform.

    Parameters explained:
        rho           — grid resolution in pixels (1 = finest)
        theta         — angle resolution in radians
        threshold     — minimum votes to count as a line
        min_line_length — shortest segment to keep (filters noise)
        max_line_gap  — max gap between segments to still join them

    Args:
        roi_edges: edge image with ROI applied (from Day 3)
        cfg: hough section from config YAML

    Returns:
        Array of line segments shape (N, 1, 4) where each row is
        [x1, y1, x2, y2], or None if no lines detected
    """
    theta = np.deg2rad(cfg["theta_degrees"])

    lines = cv2.HoughLinesP(
        roi_edges,
        rho=cfg["rho"],
        theta=theta,
        threshold=cfg["threshold"],
        minLineLength=cfg["min_line_length"],
        maxLineGap=cfg["max_line_gap"],
    )

    return lines


def draw_raw_lines(frame: np.ndarray, lines) -> np.ndarray:
    """
    Draw all detected Hough line segments directly onto the frame.
    This is the raw unfiltered output — useful for debugging.

    Each line is drawn in red so it's clearly visible.

    Args:
        frame: original BGR frame
        lines: output from detect_hough_lines

    Returns:
        Frame with all raw detected lines drawn on it
    """
    output = frame.copy()

    if lines is None:
        return output

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return output


def get_line_slope_intercept(line):
    """
    Calculate slope and intercept of a line segment.

    Used in Day 5 to separate left and right lanes by slope sign:
        negative slope → left lane (goes up-left in image coords)
        positive slope → right lane (goes up-right in image coords)

    Args:
        line: single line [x1, y1, x2, y2]

    Returns:
        (slope, intercept) tuple or None if line is vertical
    """
    x1, y1, x2, y2 = line[0]

    # Avoid division by zero for perfectly vertical lines
    if x2 - x1 == 0:
        return None

    slope     = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    return slope, intercept