import cv2
import yaml
import argparse
import numpy as np
from src.video_io import (
    get_video_capture,
    get_video_properties,
    get_video_writer,
    release_all,
)
from src.color_mask import get_lane_mask, combine_masks
from src.edge_detection import apply_canny, get_roi_vertices, apply_roi
from src.hough import detect_hough_lines, draw_raw_lines
from src.lane_lines import (
    separate_lines,
    weighted_average_line,
    extrapolate_line,
    draw_lane_lines,
)
from src.perspective import (
    get_perspective_transform,
    warp_to_bev,
    draw_source_trapezoid,
)
from src.lane_fit import (
    detect_lane_pixels,
    fit_polynomial,
    generate_lane_points,
    draw_bev_lanes,
    project_lanes_to_camera,
)
from src.smoother import LaneSmoother


# ── Stage registry ──────────────────────────────────────────────────────────
STAGES = {
    1: {"name": "Raw Passthrough",  "suffix": "day1_raw_passthrough"},
    2: {"name": "Color Mask",       "suffix": "day2_color_mask"},
    3: {"name": "Edges + ROI",      "suffix": "day3_edges_roi"},
    4: {"name": "Hough Lines",      "suffix": "day4_hough_lines"},
    5: {"name": "Lane Overlay",     "suffix": "day5_lane_overlay"},
    6: {"name": "BEV Transform",    "suffix": "day6_bev_transform"},
    7: {"name": "Poly Fit",         "suffix": "day7_poly_fit"},
    8: {"name": "Full Pipeline",    "suffix": "day8_full_pipeline"},
}


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Lane Detection Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=8,
        choices=list(STAGES.keys()),
        help=(
            "Pipeline stage to run:\n"
            "  1 = Raw Passthrough\n"
            "  2 = Color Mask\n"
            "  3 = Edges + ROI\n"
            "  4 = Hough Lines\n"
            "  5 = Lane Overlay\n"
            "  6 = BEV Transform\n"
            "  7 = Poly Fit\n"
            "  8 = Full Pipeline\n"
        ),
    )
    return parser.parse_args()


def build_output_path(base_path: str, stage: int) -> str:
    suffix = STAGES[stage]["suffix"]
    folder = base_path.rsplit("/", 1)[0]
    return f"{folder}/{suffix}.mp4"


def add_info_overlay(frame: np.ndarray, stage: int,
                     frame_count: int, fps: float) -> np.ndarray:
    """
    Add a small HUD overlay showing pipeline info on the frame.
    Makes the final output look polished and portfolio-ready.
    """
    output = frame.copy()
    h, w   = output.shape[:2]

    # Semi-transparent dark bar at top
    bar          = output.copy()
    bar[:40, :] = (20, 20, 20)
    cv2.addWeighted(bar, 0.6, output, 0.4, 0, output)

    # Text on the bar
    cv2.putText(
        output,
        f"Lane Detection Pipeline  |  Stage {stage}  |  "
        f"Frame {frame_count}  |  {fps:.0f} FPS target",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return output


def process_frame(frame, stage: int, cfg: dict,
                  roi_vertices, height: int, width: int,
                  M, Minv, smoother: LaneSmoother,
                  frame_count: int, fps: float):
    """
    Apply processing up to and including the requested stage.
    Each stage builds on the previous one.
    """
    # Stage 1 — raw frame
    if stage == 1:
        return frame

    # Stage 2 — color mask
    mask = get_lane_mask(frame, cfg["color_mask"])
    if stage == 2:
        return combine_masks(frame, mask)

    # Stage 3 — edges + ROI
    edges     = apply_canny(mask, cfg["canny"])
    roi_edges = apply_roi(edges, roi_vertices)
    if stage == 3:
        output = frame.copy()
        output[roi_edges > 0] = [0, 255, 0]
        return output

    # Stage 4 — raw Hough lines
    lines = detect_hough_lines(roi_edges, cfg["hough"])
    if stage == 4:
        return draw_raw_lines(frame, lines)

    # Stage 5 — clean Hough lane overlay
    left_raw, right_raw = separate_lines(lines, cfg["lane_lines"])
    left_avg            = weighted_average_line(left_raw)
    right_avg           = weighted_average_line(right_raw)

    y_bottom     = height
    y_top        = int(height * cfg["roi"]["top_left_y"])
    left_coords  = extrapolate_line(left_avg,  y_bottom, y_top)
    right_coords = extrapolate_line(right_avg, y_bottom, y_top)

    if stage == 5:
        return draw_lane_lines(frame, left_coords, right_coords,
                               cfg["lane_lines"])

    # Stage 6 — BEV transform
    bev = warp_to_bev(mask, M, width, height)
    if stage == 6:
        debug_frame  = draw_source_trapezoid(
            frame, height, width, cfg["perspective"]
        )
        bev_bgr      = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
        side_by_side = np.hstack([debug_frame, bev_bgr])
        return cv2.resize(side_by_side, (width, height))

    # Stage 7 — polynomial fit in BEV
    bev = warp_to_bev(mask, M, width, height)

    left_x, left_y, right_x, right_y = detect_lane_pixels(bev)
    left_coeffs  = fit_polynomial(left_x,  left_y)
    right_coeffs = fit_polynomial(right_x, right_y)

    if stage == 7:
        left_pts  = generate_lane_points(left_coeffs,  height)
        right_pts = generate_lane_points(right_coeffs, height)
        bev_drawn = draw_bev_lanes(bev, left_pts, right_pts)

        cam_overlay = project_lanes_to_camera(
            frame, bev, left_coeffs, right_coeffs,
            Minv, height, width, cfg["lane_lines"]
        )
        side_by_side = np.hstack([bev_drawn, cam_overlay])
        return cv2.resize(side_by_side, (width, height))

    # Stage 8 — full pipeline with temporal smoothing + HUD
    bev = warp_to_bev(mask, M, width, height)

    left_x, left_y, right_x, right_y = detect_lane_pixels(bev)
    left_coeffs  = fit_polynomial(left_x,  left_y)
    right_coeffs = fit_polynomial(right_x, right_y)

    # Apply temporal smoothing — fixes dashed line flickering
    left_smooth, right_smooth = smoother.update(left_coeffs, right_coeffs)

    # Project smoothed lanes back to camera view
    output = project_lanes_to_camera(
        frame, bev, left_smooth, right_smooth,
        Minv, height, width, cfg["lane_lines"]
    )

    # Add HUD info overlay
    output = add_info_overlay(output, stage, frame_count, fps)

    return output


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    stage      = args.stage
    stage_name = STAGES[stage]["name"]
    out_path   = build_output_path(cfg["video"]["output_path"], stage)

    print(f"\n{'='*50}")
    print(f"  Stage {stage}: {stage_name}")
    print(f"  Input : {cfg['video']['input_path']}")
    print(f"  Output: {out_path}")
    print(f"{'='*50}\n")

    cap    = get_video_capture(cfg["video"]["input_path"])
    props  = get_video_properties(cap)
    height = props["height"]
    width  = props["width"]
    fps    = props["fps"]

    writer = get_video_writer(out_path, fps, width, height)

    print(f"Video: {width}x{height} "
          f"@ {fps:.1f}fps | {props['total_frames']} frames\n")

    roi_vertices = get_roi_vertices(height, width, cfg["roi"])
    M, Minv      = get_perspective_transform(
        height, width, cfg["perspective"]
    )

    # Initialise smoother — one instance persists across all frames
    smoother = LaneSmoother(
        alpha=cfg["smoother"]["alpha"],
        max_age=cfg["smoother"]["max_age"],
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = process_frame(
            frame, stage, cfg, roi_vertices,
            height, width, M, Minv,
            smoother, frame_count, fps
        )
        writer.write(output_frame)
        frame_count += 1

        if cfg["video"]["display"]:
            cv2.imshow(f"Stage {stage}: {stage_name}", output_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    release_all(cap, writer)
    cv2.destroyAllWindows()
    print(f"Done. {frame_count} frames → {out_path}")


if __name__ == "__main__":
    main()