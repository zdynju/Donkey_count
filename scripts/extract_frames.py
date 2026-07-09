#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video_io import iter_video_frames, parse_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract representative frames from video.")
    parser.add_argument("--source", required=True, help="Input video path.")
    parser.add_argument("--output-dir", default="data/raw/frames", help="Frame output directory.")
    parser.add_argument("--start", default=None, help="Start time: seconds, MM:SS, or HH:MM:SS.")
    parser.add_argument("--end", default=None, help="End time: seconds, MM:SS, or HH:MM:SS.")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Frames extracted per second.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import cv2

    source = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for frame_index, timestamp_sec, frame in iter_video_frames(
        source,
        start_sec=parse_time(args.start),
        end_sec=parse_time(args.end),
        sample_fps=args.sample_fps,
    ):
        output_path = output_dir / f"{source.stem}_{frame_index:08d}_{timestamp_sec:.2f}s.jpg"
        cv2.imwrite(str(output_path), frame)
        written += 1

    print(f"frames_written={written}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
