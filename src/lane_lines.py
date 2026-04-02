import cv2
import numpy as np


def separate_lines(lines, cfg: dict):
    """
    Split raw Hough lines into left and right lane candidates.

    Logic:
        negative slope → left lane  (line rises toward left)
        positive slope → right lane (line rises toward right)

    Lines with slopes too shallow (noise) or too steep (vertical
    artifacts) are discarded using the slope filter range.

    Args:
        lines: raw output from detect_hough_lines (can be None)
        cfg: lane_lines section from config YAML

    Returns:
        left_lines, right_lines — two lists of (slope, intercept, length)
    """
    left_lines  = []
    right_lines = []

    if lines is None:
        return left_lines, right_lines

    min_slope = cfg["min_slope"]
    max_slope = cfg["max_slope"]

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Skip perfectly vertical lines
        if x2 - x1 == 0:
            continue

        slope     = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        length    = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        # Filter out shallow and overly steep lines
        if abs(slope) < min_slope or abs(slope) > max_slope:
            continue

        if slope < 0:
            left_lines.append((slope, intercept, length))
        else:
            right_lines.append((slope, intercept, length))

    return left_lines, right_lines


def weighted_average_line(lines):
    """
    Compute a single slope + intercept from multiple line candidates.

    Uses line length as weight — longer segments are more reliable
    and should influence the final line position more than short ones.

    Args:
        lines: list of (slope, intercept, length) tuples

    Returns:
        (slope, intercept) averaged tuple, or None if no lines
    """
    if not lines:
        return None

    slopes     = np.array([l[0] for l in lines])
    intercepts = np.array([l[1] for l in lines])
    lengths    = np.array([l[2] for l in lines])

    total_length = lengths.sum()
    if total_length == 0:
        return None

    avg_slope     = np.dot(slopes, lengths)     / total_length
    avg_intercept = np.dot(intercepts, lengths) / total_length

    return avg_slope, avg_intercept


def extrapolate_line(slope_intercept, y_start: int, y_end: int):
    """
    Convert slope + intercept into pixel coordinates.

    Given slope m and intercept b from y = mx + b:
        x = (y - b) / m

    We calculate x at the bottom of frame (y_start) and top of
    ROI (y_end) to get the full visible lane line.

    Args:
        slope_intercept: (slope, intercept) tuple or None
        y_start: bottom y coordinate (bottom of frame)
        y_end: top y coordinate (top of ROI)

    Returns:
        (x1, y1, x2, y2) pixel coordinates, or None if input is None
    """
    if slope_intercept is None:
        return None

    slope, intercept = slope_intercept

    # Avoid division by zero
    if slope == 0:
        return None

    x1 = int((y_start - intercept) / slope)
    x2 = int((y_end   - intercept) / slope)

    return x1, y_start, x2, y_end


def draw_lane_lines(frame: np.ndarray, left_coords, right_coords,
                    cfg: dict) -> np.ndarray:
    """
    Draw the two final lane lines and a filled polygon between them.

    The filled polygon gives the satisfying green shaded lane area
    that makes the output look like a real ADAS system.

    Args:
        frame: original BGR frame
        left_coords: (x1,y1,x2,y2) for left lane or None
        right_coords: (x1,y1,x2,y2) for right lane or None
        cfg: lane_lines section from config YAML

    Returns:
        Frame with lane lines and filled polygon overlaid
    """
    output = frame.copy()
    overlay = np.zeros_like(frame)

    # Draw filled polygon between lanes if both detected
    if left_coords is not None and right_coords is not None:
        lx1, ly1, lx2, ly2 = left_coords
        rx1, ry1, rx2, ry2 = right_coords

        # Polygon vertices: bottom-left, top-left, top-right, bottom-right
        polygon = np.array([[
            (lx1, ly1),
            (lx2, ly2),
            (rx2, ry2),
            (rx1, ry1),
        ]], dtype=np.int32)

        cv2.fillPoly(overlay, polygon, cfg["fill_color"])

        # Blend overlay with original frame (transparency effect)
        alpha = cfg["fill_alpha"]
        output = cv2.addWeighted(output, 1.0, overlay, alpha, 0)

    # Draw left lane line
    if left_coords is not None:
        lx1, ly1, lx2, ly2 = left_coords
        cv2.line(output, (lx1, ly1), (lx2, ly2),
                 cfg["line_color"], cfg["line_thickness"])

    # Draw right lane line
    if right_coords is not None:
        rx1, ry1, rx2, ry2 = right_coords
        cv2.line(output, (rx1, ry1), (rx2, ry2),
                 cfg["line_color"], cfg["line_thickness"])

    return output