from __future__ import annotations

from typing import Iterable

import numpy as np

from .detector import Detection


def draw_detections(
    image: np.ndarray,
    detections: Iterable[Detection],
    count: int | None = None,
) -> np.ndarray:
    import cv2

    output = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection.bbox]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 180, 0), 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )

    if count is not None:
        cv2.putText(
            output,
            f"donkey count: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return output
