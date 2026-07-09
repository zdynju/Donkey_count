from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np


def parse_time(value: str | None) -> float | None:
    if value is None:
        return None

    parts = value.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    raise ValueError(f"Invalid time value: {value}. Use seconds, MM:SS, or HH:MM:SS.")


def iter_video_frames(
    source: str | Path,
    start_sec: float | None = None,
    end_sec: float | None = None,
    sample_fps: float | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    import cv2

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {source}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int((start_sec or 0) * native_fps)
    end_frame = int(end_sec * native_fps) if end_sec is not None else None
    step = max(1, int(round(native_fps / sample_fps))) if sample_fps else 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    try:
        while True:
            frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if end_frame is not None and frame_index > end_frame:
                break

            ok, frame = cap.read()
            if not ok:
                break

            if (frame_index - start_frame) % step == 0:
                yield frame_index, frame_index / native_fps, frame
    finally:
        cap.release()


def open_video_writer(
    output_path: str | Path,
    fps: float,
    frame_size: tuple[int, int],
) -> object:
    import cv2

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
