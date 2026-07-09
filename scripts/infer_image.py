#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.counter import count_detections
from src.detector import DonkeyDetector
from src.visualization import draw_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run donkey detection on one image.")
    parser.add_argument("--source", required=True, help="Input image path.")
    parser.add_argument("--weights", default="models/best.pt", help="YOLO .pt weights.")
    parser.add_argument("--output-dir", default="outputs/images", help="Output directory.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import cv2

    source = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(source))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {source}")

    detector = DonkeyDetector(args.weights, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
    detections = detector.predict(image)
    count = count_detections(detections)

    visualized = draw_detections(image, detections, count=count)
    image_output = output_dir / f"{source.stem}_detected.jpg"
    json_output = output_dir / f"{source.stem}_detections.json"

    cv2.imwrite(str(image_output), visualized)
    json_output.write_text(
        json.dumps(
            {
                "source": str(source),
                "count": count,
                "detections": [detection.to_dict() for detection in detections],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"count={count}")
    print(f"image_output={image_output}")
    print(f"json_output={json_output}")


if __name__ == "__main__":
    main()
