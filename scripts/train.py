#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the donkey detector with Ultralytics YOLO.")
    parser.add_argument("--config", default="configs/train.yaml", help="Training config YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'ultralytics'. Install project dependencies with "
            "'pip install -r requirements.txt'."
        ) from exc

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = YOLO(config["model"])
    model.train(
        data=config["data"],
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=config["device"],
        workers=config["workers"],
        patience=config["patience"],
        project="outputs/training",
        name="donkey_detector",
    )


if __name__ == "__main__":
    main()

