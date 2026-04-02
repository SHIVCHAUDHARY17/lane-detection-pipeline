import cv2
import numpy as np


def get_perspective_transform(height: int, width: int, cfg: dict):
    """
    Compute the perspective transform matrix and its inverse.

    We define four source points on the road in the camera view
    (a trapezoid shape following the lane) and four destination
    points in the output bird's eye view (a rectangle).

    OpenCV then computes the homography matrix M that maps every
    pixel from camera view → top-down view.

    The inverse matrix Minv maps back: top-down → camera view.
    We need Minv later to project BEV lane fits back onto the
    original frame.

    Args:
        height: frame height in pixels
        width: frame width in pixels
        cfg: perspective section from config YAML

    Returns:
        M (transform matrix), Minv (inverse transform matrix)
    """
    # Source points — trapezoid on the road in camera view
    # These follow the lane shape as it converges to vanishing point
    src = np.float32([
        [width * cfg["src_top_left_x"],     height * cfg["src_top_y"]],
        [width * cfg["src_top_right_x"],    height * cfg["src_top_y"]],
        [width * cfg["src_bottom_right_x"], height * cfg["src_bottom_y"]],
        [width * cfg["src_bottom_left_x"],  height * cfg["src_bottom_y"]],
    ])

    # Destination points — rectangle in bird's eye view
    # Lanes will appear parallel and straight in this view
    dst = np.float32([
        [width * cfg["dst_left_x"],  0],
        [width * cfg["dst_right_x"], 0],
        [width * cfg["dst_right_x"], height],
        [width * cfg["dst_left_x"],  height],
    ])

    M    = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv


def warp_to_bev(image: np.ndarray, M: np.ndarray,
                width: int, height: int) -> np.ndarray:
    """
    Apply perspective transform to get bird's eye view.

    cv2.warpPerspective applies the homography matrix M to every
    pixel — remapping them from camera coordinates to top-down
    coordinates.

    INTER_LINEAR interpolation fills gaps smoothly.
    BORDER_CONSTANT fills any border areas with black.

    Args:
        image: input frame or mask (BGR or grayscale)
        M: perspective transform matrix from get_perspective_transform
        width: output width in pixels
        height: output height in pixels

    Returns:
        Bird's eye view image same size as input
    """
    return cv2.warpPerspective(
        image, M, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


def draw_source_trapezoid(frame: np.ndarray, height: int,
                          width: int, cfg: dict) -> np.ndarray:
    """
    Draw the source trapezoid on the original frame for debugging.
    Useful to verify your perspective points are placed correctly
    before running the full transform.

    Args:
        frame: original BGR frame
        height: frame height
        width: frame width
        cfg: perspective section from config YAML

    Returns:
        Frame with trapezoid drawn on it
    """
    src = np.int32([
        [width * cfg["src_top_left_x"],     height * cfg["src_top_y"]],
        [width * cfg["src_top_right_x"],    height * cfg["src_top_y"]],
        [width * cfg["src_bottom_right_x"], height * cfg["src_bottom_y"]],
        [width * cfg["src_bottom_left_x"],  height * cfg["src_bottom_y"]],
    ])

    output = frame.copy()
    cv2.polylines(output, [src], isClosed=True,
                  color=(0, 255, 255), thickness=2)

    # Label the corners for clarity
    labels = ["TL", "TR", "BR", "BL"]
    for point, label in zip(src, labels):
        cv2.circle(output, tuple(point), 6, (0, 0, 255), -1)
        cv2.putText(output, label, tuple(point + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return output