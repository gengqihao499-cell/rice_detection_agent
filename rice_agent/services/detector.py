from __future__ import annotations

import time
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from ultralytics import YOLO

from rice_agent.config import settings
from rice_agent.domain import metadata_for_class, validate_model_names


SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


class RiceDiseaseDetector:
    """YOLO11L 水稻病虫害推理服务。"""

    def __init__(
        self,
        model_path: Path | str | None = None,
        device: str | int | None = None,
    ) -> None:
        self.model_path = Path(
            model_path or settings.yolo_model_path
        ).expanduser().resolve()

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"找不到YOLO权重：{self.model_path}"
            )

        self.device = self._resolve_device(
            settings.yolo_device if device is None else device
        )
        self._model: YOLO | None = None
        self._load_lock = Lock()
        self._predict_lock = Lock()
        self.model_name_warnings: list[str] = []

    @staticmethod
    def _resolve_device(value: str | int) -> str | int:
        if isinstance(value, int):
            return value

        normalized = str(value).strip().lower()

        if normalized == "auto":
            return 0 if torch.cuda.is_available() else "cpu"

        if normalized.isdigit():
            return int(normalized)

        return normalized

    @property
    def model(self) -> YOLO:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    self._model = YOLO(str(self.model_path))
                    self.model_name_warnings = validate_model_names(
                        self._model.names
                    )
        return self._model

    def model_info(self) -> dict[str, Any]:
        model = self.model
        names = {
            int(key): str(value)
            for key, value in model.names.items()
        }

        return {
            "model_path": str(self.model_path),
            "device": str(self.device),
            "class_names": names,
            "class_count": len(names),
            "warnings": self.model_name_warnings,
        }

    def detect(
        self,
        image_path: str | Path,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
        image_size: int | None = None,
        save_annotated: bool = True,
    ) -> dict[str, Any]:
        path = Path(image_path).expanduser().resolve()

        if not path.is_file():
            return {
                "success": False,
                "error": "image_not_found",
                "message": f"图片不存在：{path}",
            }

        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            return {
                "success": False,
                "error": "unsupported_image_type",
                "message": f"不支持的图片格式：{path.suffix}",
            }

        conf = (
            settings.yolo_confidence
            if confidence_threshold is None
            else confidence_threshold
        )
        iou = settings.yolo_iou if iou_threshold is None else iou_threshold
        imgsz = settings.yolo_image_size if image_size is None else image_size

        if not 0.0 <= conf <= 1.0:
            raise ValueError("confidence_threshold必须在0到1之间")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou_threshold必须在0到1之间")

        started_at = time.perf_counter()

        # 同一模型实例串行推理，避免Notebook/Gradio并发时争用GPU状态。
        with self._predict_lock:
            results = self.model.predict(
                source=str(path),
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=self.device,
                verbose=False,
            )

        result = results[0]
        image_height, image_width = result.orig_shape

        detections: list[dict[str, Any]] = []
        class_summary: dict[str, dict[str, Any]] = {}

        if result.boxes is not None:
            xyxy_values = result.boxes.xyxy.detach().cpu().tolist()
            score_values = result.boxes.conf.detach().cpu().tolist()
            class_values = result.boxes.cls.detach().cpu().tolist()

            for bbox, score, class_value in zip(
                xyxy_values,
                score_values,
                class_values,
            ):
                class_id = int(class_value)
                raw_name = str(result.names.get(class_id, class_id))
                metadata = metadata_for_class(class_id, raw_name)

                item = {
                    "class_id": class_id,
                    "raw_class_name": raw_name,
                    "disease_code": metadata["code"],
                    "class_name_zh": metadata["display_name_zh"],
                    "kind": metadata["kind"],
                    "confidence": round(float(score), 4),
                    "bbox": {
                        "x1": round(float(bbox[0]), 2),
                        "y1": round(float(bbox[1]), 2),
                        "x2": round(float(bbox[2]), 2),
                        "y2": round(float(bbox[3]), 2),
                    },
                }
                detections.append(item)

                code = metadata["code"]
                summary = class_summary.setdefault(
                    code,
                    {
                        "disease_code": code,
                        "class_name_zh": metadata["display_name_zh"],
                        "raw_class_name": raw_name,
                        "kind": metadata["kind"],
                        "count": 0,
                        "max_confidence": 0.0,
                        "mean_confidence": 0.0,
                        "_confidence_sum": 0.0,
                    },
                )
                summary["count"] += 1
                summary["max_confidence"] = max(
                    summary["max_confidence"],
                    float(score),
                )
                summary["_confidence_sum"] += float(score)

        summaries: list[dict[str, Any]] = []

        for summary in class_summary.values():
            count = int(summary["count"])
            summary["mean_confidence"] = round(
                summary["_confidence_sum"] / count,
                4,
            )
            summary["max_confidence"] = round(
                summary["max_confidence"],
                4,
            )
            summary.pop("_confidence_sum", None)
            summaries.append(summary)

        summaries.sort(
            key=lambda value: value["max_confidence"],
            reverse=True,
        )

        abnormal = [
            item
            for item in summaries
            if item["kind"] in {"disease", "pest"}
        ]
        healthy = [
            item for item in summaries if item["kind"] == "healthy"
        ]

        primary_result = (
            abnormal[0]
            if abnormal
            else summaries[0] if summaries else None
        )
        health_conflict = bool(abnormal and healthy)

        annotated_image_path: str | None = None

        if save_annotated:
            settings.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = (
                settings.output_dir
                / f"{path.stem}_detected.jpg"
            )
            result.save(filename=str(output_path))
            annotated_image_path = str(output_path)

        elapsed_ms = round(
            (time.perf_counter() - started_at) * 1000,
            2,
        )

        return {
            "success": True,
            "mock": False,
            "image_path": str(path),
            "annotated_image_path": annotated_image_path,
            "model": {
                "name": "YOLO11L-Rice-Disease-Detection",
                "path": str(self.model_path),
                "device": str(self.device),
                "image_size": imgsz,
                "class_name_warnings": self.model_name_warnings,
            },
            "confidence_threshold": conf,
            "iou_threshold": iou,
            "image_width": image_width,
            "image_height": image_height,
            "detection_count": len(detections),
            "primary_result": primary_result,
            "class_summary": summaries,
            "health_conflict": health_conflict,
            "detections": detections,
            "latency_ms": elapsed_ms,
            "notice": (
                "检测结果只用于辅助筛查，不构成确定诊断。"
                "类别范围受公开模型训练数据限制。"
            ),
        }
