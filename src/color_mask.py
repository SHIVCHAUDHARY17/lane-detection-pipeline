import cv2
import numpy as np


def apply_white_mask(frame: np.ndarray, threshold: int) -> np.ndarray:
    """
    Isolate white lane markings using grayscale brightness threshold.

    White pixels are bright in all channels, so grayscale works well.
    Any pixel brighter than `threshold` is kept, rest is black.

    Args:
        frame: BGR image (standard OpenCV format)
        threshold: brightness cutoff (0-255), typically 190-210

    Returns:
        Binary mask — white where lane likely is, black elsewhere
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return white_mask


def apply_yellow_mask(
    frame: np.ndarray,
    lower: list,
    upper: list
) -> np.ndarray:
    """
    Isolate yellow lane markings using HLS color space thresholding.

    HLS separates hue from lightness, making yellow detection more
    robust under different lighting conditions than RGB would be.

    Args:
        frame: BGR image
        lower: HLS lower bound [H, L, S] — e.g. [15, 100, 100]
        upper: HLS upper bound [H, L, S] — e.g. [35, 255, 255]

    Returns:
        Binary mask — white where yellow lane likely is, black elsewhere
    """
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    lower_bound = np.array(lower, dtype=np.uint8)
    upper_bound = np.array(upper, dtype=np.uint8)
    yellow_mask = cv2.inRange(hls, lower_bound, upper_bound)
    return yellow_mask


def combine_masks(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a binary mask to the original frame.
    Only pixels where mask is white are kept — rest goes black.

    Args:
        frame: original BGR frame
        mask: binary mask (same height/width as frame)

    Returns:
        Masked BGR frame showing only pixels of interest
    """
    return cv2.bitwise_and(frame, frame, mask=mask)


def get_lane_mask(frame: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Main function — combines white and yellow masks into one lane mask.

    This is what the pipeline will call. It:
    1. Detects white pixels via grayscale threshold
    2. Detects yellow pixels via HLS range
    3. Merges both into a single binary mask

    Args:
        frame: BGR video frame
        cfg: color_mask section from config YAML

    Returns:
        Combined binary mask covering both white and yellow lanes
    """
    white_mask = apply_white_mask(frame, cfg["white_threshold"])

    yellow_mask = apply_yellow_mask(
        frame,
        cfg["hls_yellow_lower"],
        cfg["hls_yellow_upper"]
    )

    # OR the two masks — keep pixels that are white OR yellow
    combined = cv2.bitwise_or(white_mask, yellow_mask)
    return combined