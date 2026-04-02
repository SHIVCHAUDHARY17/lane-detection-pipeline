import cv2
import numpy as np


def detect_lane_pixels(bev_mask: np.ndarray):
    """
    Find all white pixel coordinates in the BEV mask.

    Instead of Hough lines, we now work directly with pixel
    coordinates in BEV space. White pixels = lane markings.

    We split pixels into left and right halves by x position —
    left half of image = left lane, right half = right lane.

    Args:
        bev_mask: binary mask warped to bird's eye view

    Returns:
        left_x, left_y, right_x, right_y — pixel coordinate arrays
    """
    height, width = bev_mask.shape

    # Find all white pixel coordinates
    # nonzero() returns (row_indices, col_indices) = (y, x)
    y_pixels, x_pixels = np.nonzero(bev_mask)

    midpoint = width // 2

    # Split into left and right by x position
    left_mask  = x_pixels < midpoint
    right_mask = x_pixels >= midpoint

    left_x  = x_pixels[left_mask]
    left_y  = y_pixels[left_mask]
    right_x = x_pixels[right_mask]
    right_y = y_pixels[right_mask]

    return left_x, left_y, right_x, right_y


def fit_polynomial(x_pixels: np.ndarray,
                   y_pixels: np.ndarray, degree: int = 2):
    """
    Fit a polynomial curve through lane pixel coordinates.

    We fit x as a function of y (not y as function of x) because
    lanes are nearly vertical in BEV — fitting y=f(x) would be
    numerically unstable for near-vertical lines.

    np.polyfit returns coefficients [a, b, c] for:
        x = a*y^2 + b*y + c

    Args:
        x_pixels: x coordinates of lane pixels
        y_pixels: y coordinates of lane pixels
        degree: polynomial degree (2 = quadratic, handles curves)

    Returns:
        Polynomial coefficients array, or None if too few points
    """
    # Need minimum points for a reliable fit
    if len(x_pixels) < 10:
        return None

    try:
        coeffs = np.polyfit(y_pixels, x_pixels, degree)
        return coeffs
    except np.RankWarning:
        return None


def generate_lane_points(coeffs, height: int):
    """
    Generate (x, y) points along the fitted polynomial curve.

    For every y value from top to bottom of frame, compute
    the corresponding x using x = a*y^2 + b*y + c.

    This gives us a smooth curve we can draw as a polyline.

    Args:
        coeffs: polynomial coefficients from fit_polynomial
        height: frame height — defines y range

    Returns:
        Array of (x, y) integer points for cv2.polylines
    """
    if coeffs is None:
        return None

    y_vals = np.linspace(0, height - 1, height)
    x_vals = np.polyval(coeffs, y_vals)

    # Stack into (N, 1, 2) shape required by cv2.polylines
    points = np.array(
        list(zip(x_vals.astype(int), y_vals.astype(int)))
    ).reshape(-1, 1, 2)

    return points


def draw_bev_lanes(bev_mask: np.ndarray, left_points,
                   right_points) -> np.ndarray:
    """
    Draw fitted polynomial curves on the BEV image.

    Converts grayscale BEV mask to BGR so we can draw
    colored curves on top of the white lane pixels.

    Left lane = blue, Right lane = red, so they are
    visually distinct from the white pixel background.

    Args:
        bev_mask: binary BEV mask
        left_points: curve points for left lane
        right_points: curve points for right lane

    Returns:
        BGR image with fitted curves drawn
    """
    output = cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)

    if left_points is not None:
        cv2.polylines(output, [left_points],
                      isClosed=False, color=(255, 0, 0), thickness=4)

    if right_points is not None:
        cv2.polylines(output, [right_points],
                      isClosed=False, color=(0, 0, 255), thickness=4)

    return output


def project_lanes_to_camera(frame: np.ndarray, bev_mask: np.ndarray,
                             left_coeffs, right_coeffs,
                             Minv: np.ndarray, height: int,
                             width: int, cfg: dict) -> np.ndarray:
    """
    Project polynomial lane fits back onto the original camera frame.

    Steps:
    1. Draw filled polygon between lanes in BEV space
    2. Warp that polygon back to camera view using Minv
    3. Blend with original frame

    This gives the final satisfying lane overlay using
    polynomial curves rather than straight Hough lines.

    Args:
        frame: original BGR camera frame
        bev_mask: binary BEV mask
        left_coeffs: left lane polynomial coefficients
        right_coeffs: right lane polynomial coefficients
        Minv: inverse perspective transform matrix
        height: frame height
        width: frame width
        cfg: lane_lines section from config YAML

    Returns:
        Camera frame with polynomial lane overlay
    """
    # Create blank BEV canvas to draw lane polygon
    lane_canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if left_coeffs is not None and right_coeffs is not None:
        y_vals     = np.linspace(0, height - 1, height)
        left_x     = np.polyval(left_coeffs,  y_vals)
        right_x    = np.polyval(right_coeffs, y_vals)

        # Build polygon: left curve top→bottom, right curve bottom→top
        left_pts   = np.array([left_x,  y_vals]).T
        right_pts  = np.array([right_x, y_vals]).T[::-1]
        polygon    = np.vstack([left_pts, right_pts]).astype(np.int32)

        cv2.fillPoly(lane_canvas, [polygon], cfg["fill_color"])

    # Warp filled polygon back to camera perspective using Minv
    lane_warped = cv2.warpPerspective(lane_canvas, Minv, (width, height))

    # Blend with original frame
    alpha  = cfg["fill_alpha"]
    output = cv2.addWeighted(frame, 1.0, lane_warped, alpha, 0)

    return output