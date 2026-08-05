from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _resolve_project_path(value: str, default: Path) -> Path:
    raw = Path(value).expanduser() if value else default
    if not raw.is_absolute():
        raw = PROJECT_ROOT / raw
    return raw.resolve()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path = PROJECT_ROOT

    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com",
    )
    deepseek_model: str = os.getenv(
        "DEEPSEEK_MODEL",
        "deepseek-chat",
    )

    yolo_model_path: Path = _resolve_project_path(
        os.getenv("RICE_YOLO_MODEL_PATH", ""),
        PROJECT_ROOT / "models" / "YOLO11L-Rice-Disease-Detection.pt",
    )
    yolo_device: str = os.getenv("YOLO_DEVICE", "auto")
    yolo_image_size: int = int(os.getenv("YOLO_IMAGE_SIZE", "640"))
    yolo_confidence: float = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
    yolo_iou: float = float(os.getenv("YOLO_IOU", "0.45"))
    max_detection_boxes_for_llm: int = int(
        os.getenv("MAX_DETECTION_BOXES_FOR_LLM", "80")
    )

    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "BAAI/bge-small-zh-v1.5",
    )
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    rag_chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "500"))
    rag_chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
    rag_precise_threshold: float = float(
        os.getenv("RAG_PRECISE_THRESHOLD", "0.78")
    )
    rag_reference_threshold: float = float(
        os.getenv("RAG_REFERENCE_THRESHOLD", "0.45")
    )
    rag_min_precise_chunks: int = int(
        os.getenv("RAG_MIN_PRECISE_CHUNKS", "1")
    )

    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "10"))
    max_tool_result_chars: int = int(
        os.getenv("MAX_TOOL_RESULT_CHARS", "30000")
    )
    chat_window_turns: int = int(os.getenv("CHAT_WINDOW_TURNS", "6"))
    chat_max_history_chars: int = int(
        os.getenv("CHAT_MAX_HISTORY_CHARS", "12000")
    )
    hallucination_target: float = float(
        os.getenv("HALLUCINATION_TARGET", "0.90")
    )
    hallucination_max_retries: int = int(
        os.getenv("HALLUCINATION_MAX_RETRIES", "2")
    )

    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "8"))
    evaluation_enabled: bool = _env_bool("EVALUATION_ENABLED", True)
    evaluation_timeout_seconds: float = float(
        os.getenv("EVALUATION_TIMEOUT_SECONDS", "45")
    )

    knowledge_dir: Path = PROJECT_ROOT / "knowledge" / "rice_documents"
    chroma_dir: Path = PROJECT_ROOT / "data" / "rice_chroma_db"
    output_dir: Path = PROJECT_ROOT / "outputs"
    upload_dir: Path = PROJECT_ROOT / "uploads"
    evaluation_log_path: Path = output_dir / "evaluations.jsonl"
    feedback_log_path: Path = output_dir / "feedback.jsonl"


settings = Settings()
