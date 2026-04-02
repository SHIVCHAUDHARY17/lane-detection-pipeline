import numpy as np
import pytest
from src.edge_detection import apply_canny, get_roi_vertices, apply_roi


def make_binary_mask():
    """Simple binary mask simulating color filter output."""
    mask = np.zeros((540, 960), dtype=np.uint8)
    mask[300:400, 200:700] = 255
    return mask


def test_canny_returns_same_shape():
    """
    Canny output should have same dimensions as input mask.
    """
    mask   = make_binary_mask()
    cfg    = {"blur_kernel": 5, "low_threshold": 50, "high_threshold": 150}
    edges  = apply_canny(mask, cfg)

    assert edges.shape == mask.shape, \
        "Canny output shape does not match input shape"


def test_canny_output_is_binary():
    """
    Canny output should only contain values 0 or 255.
    """
    mask  = make_binary_mask()
    cfg   = {"blur_kernel": 5, "low_threshold": 50, "high_threshold": 150}
    edges = apply_canny(mask, cfg)

    unique_values = np.unique(edges)
    for val in unique_values:
        assert val in [0, 255], \
            f"Unexpected value {val} in Canny output — expected 0 or 255"


def test_roi_vertices_shape():
    """
    ROI vertices should have shape (1, 3, 2) for fillPoly compatibility.
    """
    cfg = {"top_left_x": 0.45, "top_left_y": 0.60,
           "top_right_x": 0.55, "top_right_y": 0.60}
    vertices = get_roi_vertices(540, 960, cfg)

    assert vertices.shape == (1, 3, 2), \
        f"Expected shape (1, 3, 2), got {vertices.shape}"


def test_roi_masks_outside_region():
    """
    After ROI mask applied, pixels outside triangle should be zero.
    """
    mask     = make_binary_mask()
    cfg      = {"blur_kernel": 5, "low_threshold": 50, "high_threshold": 150}
    edges    = apply_canny(mask, cfg)
    roi_cfg  = {"top_left_x": 0.45, "top_left_y": 0.60,
                "top_right_x": 0.55, "top_right_y": 0.60}
    vertices = get_roi_vertices(540, 960, roi_cfg)
    result   = apply_roi(edges, vertices)

    # Top of frame (above ROI) should be all zeros
    assert result[0, :].sum() == 0, \
        "Top row of frame should be masked out by ROI"