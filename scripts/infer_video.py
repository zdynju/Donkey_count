#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.counter import FrameCount, count_detections, summarize_counts
from src.detector import DonkeyDetector
from src.video_io import iter_video_frames, open_video_writer, parse_time
from src.visualization import draw_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run donkey detection on a video time range.")
    parser.add_argument("--source", required=True, help="Input video path.")
    parser.add_argument("--weights", default="models/best.pt", help="YOLO .pt weights.")
    parser.add_argument("--output-dir", default="outputs/videos", help="Output directory.")
    parser.add_argument("--report-dir", default="outputs/reports", help="Report directory.")
    parser.add_argument("--start", default=None, help="Start time: seconds, MM:SS, or HH:MM:SS.")
    parser.add_argument("--end", default=None, help="End time: seconds, MM:SS, or HH:MM:SS.")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Frames sampled per second.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--save-video", action="store_true", help="Save visualized sampled frames as video.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    detector = DonkeyDetector(args.weights, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
    start_sec = parse_time(args.start)
    end_sec = parse_time(args.end)

    writer = None
    frame_counts: list[FrameCount] = []

    try:
        for frame_index, timestamp_sec, frame in iter_video_frames(
            source,
            start_sec=start_sec,
            end_sec=end_sec,
            sample_fps=args.sample_fps,
        ):
            detections = detector.predict(frame)
            count = count_detections(detections)
            frame_counts.append(
                FrameCount(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    count=count,
                    detections=detections,
                )
            )

            if args.save_video:
                visualized = draw_detections(frame, detections, count=count)
                if writer is None:
                    height, width = visualized.shape[:2]
                    writer = open_video_writer(
                        output_dir / f"{source.stem}_detected.mp4",
                        fps=args.sample_fps,
                        frame_size=(width, height),
                    )
                writer.write(visualized)
    finally:
        if writer is not None:
            writer.release()

    summary = summarize_counts(frame_counts)
    report = {
        "source": str(source),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "sample_fps": args.sample_fps,
        "summary": summary,
        "frames": [item.to_dict() for item in frame_counts],
    }

    json_output = report_dir / f"{source.stem}_count_report.json"
    csv_output = report_dir / f"{source.stem}_count_by_time.csv"
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_output.open("w", newline="", encoding="utf-8") as file:
        writer_csv = csv.DictWriter(file, fieldnames=["frame_index", "timestamp_sec", "count"])
        writer_csv.writeheader()
        writer_csv.writerows(summary["count_by_time"])

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"json_output={json_output}")
    print(f"csv_output={csv_output}")


if __name__ == "__main__":
    main()
