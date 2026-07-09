from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: list[float]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = [round(value, 2) for value in self.bbox]
        data["confidence"] = round(self.confidence, 4)
        return data


class DonkeyDetector:
    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 960,
        class_name: str = "donkey",
    ) -> None:
        self.weights = Path(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.class_name = class_name
        self.model = self._load_model()

    def _load_model(self) -> Any:
        if not self.weights.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.weights}. "
                "Train a model first or pass --weights to an existing .pt file."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'ultralytics'. Install project dependencies with "
                "'pip install -r requirements.txt'."
            ) from exc

        return YOLO(str(self.weights))

    def predict(self, image: str | Path | np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return []
        return list(self._parse_result(results[0]))

    def _parse_result(self, result: Any) -> Iterable[Detection]:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return

        for box in boxes:
            class_id = int(box.cls[0].item())
            detected_name = str(names.get(class_id, class_id))
            if detected_name != self.class_name:
                continue

            xyxy = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
            confidence = float(box.conf[0].item())
            yield Detection(
                class_name=detected_name,
                confidence=confidence,
                bbox=xyxy,
            )

