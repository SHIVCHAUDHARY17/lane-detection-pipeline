import cv2
from pathlib import Path


def get_video_capture(input_path: str) -> cv2.VideoCapture:
    """
    Open a video file and return a VideoCapture object.
    Raises FileNotFoundError if the file does not exist.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {input_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {input_path}")

    return cap


def get_video_properties(cap: cv2.VideoCapture) -> dict:
    """
    Extract key properties from an open VideoCapture object.
    These are needed to write output video with matching settings.
    """
    return {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def get_video_writer(
    output_path: str, fps: float, width: int, height: int
) -> cv2.VideoWriter:
    """
    Create a VideoWriter to save annotated output video.
    Uses mp4v codec — widely compatible.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not create VideoWriter at: {output_path}")

    return writer


def release_all(*captures_and_writers):
    """
    Safely release any number of VideoCapture or VideoWriter objects.
    Always call this when done — otherwise the file stays locked.
    """
    for obj in captures_and_writers:
        if obj is not None:
            obj.release()