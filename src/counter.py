from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from .detector import Detection


@dataclass(frozen=True)
class FrameCount:
    frame_index: int
    timestamp_sec: float
    count: int
    detections: list[Detection]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "count": self.count,
            "detections": [detection.to_dict() for detection in self.detections],
        }


def count_detections(detections: list[Detection]) -> int:
    return len(detections)


def summarize_counts(frame_counts: list[FrameCount]) -> dict[str, Any]:
    counts = [item.count for item in frame_counts]
    if not counts:
        return {
            "frame_count": 0,
            "avg_count": 0,
            "max_count": 0,
            "min_count": 0,
            "count_by_time": [],
        }

    return {
        "frame_count": len(frame_counts),
        "avg_count": round(mean(counts), 3),
        "max_count": max(counts),
        "min_count": min(counts),
        "count_by_time": [
            {
                "frame_index": item.frame_index,
                "timestamp_sec": round(item.timestamp_sec, 3),
                "count": item.count,
            }
            for item in frame_counts
        ],
    }

