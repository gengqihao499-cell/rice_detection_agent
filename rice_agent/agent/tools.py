from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from rice_agent.config import settings
from rice_agent.services.detector import RiceDiseaseDetector
from rice_agent.services.rag_store import RiceKnowledgeStore


@lru_cache(maxsize=1)
def get_detector() -> RiceDiseaseDetector:
    return RiceDiseaseDetector()


@lru_cache(maxsize=1)
def get_knowledge_store() -> RiceKnowledgeStore:
    return RiceKnowledgeStore()


class RiceDetectionInput(BaseModel):
    image_path: str = Field(
        description="需要检测的本地水稻图片路径"
    )
    confidence_threshold: float = Field(
        default=settings.yolo_confidence,
        ge=0.0,
        le=1.0,
        description="检测置信度阈值",
    )
    iou_threshold: float = Field(
        default=settings.yolo_iou,
        ge=0.0,
        le=1.0,
        description="NMS IoU阈值",
    )


@tool(
    "detect_rice_disease",
    args_schema=RiceDetectionInput,
)
def detect_rice_disease(
    image_path: str,
    confidence_threshold: float = settings.yolo_confidence,
    iou_threshold: float = settings.yolo_iou,
) -> dict[str, Any]:
    """检测水稻病害、虫害及健康目标，并返回检测框和类别汇总。"""
    result = get_detector().detect(
        image_path=image_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        save_annotated=True,
    )

    if result.get("success"):
        detections = result.get("detections", [])
        limit = settings.max_detection_boxes_for_llm

        if len(detections) > limit:
            result["detections"] = detections[:limit]
            result["detections_truncated"] = True
            result["original_detection_count"] = len(detections)
        else:
            result["detections_truncated"] = False

    return result


class RiceKnowledgeSearchInput(BaseModel):
    question: str = Field(
        description="需要从水稻知识库检索的完整问题"
    )
    disease_code: str | None = Field(
        default=None,
        description=(
            "检测工具返回的稳定类别代码，例如leaf_blast。"
            "已知类别时必须传入。"
        ),
    )
    top_k: int = Field(
        default=settings.rag_top_k,
        ge=1,
        le=12,
        description="最多返回的知识块数量",
    )


@tool(
    "search_rice_knowledge",
    args_schema=RiceKnowledgeSearchInput,
)
def search_rice_knowledge(
    question: str,
    disease_code: str | None = None,
    top_k: int = settings.rag_top_k,
) -> dict[str, Any]:
    """从本地Chroma知识库检索水稻病虫害资料。"""
    try:
        results = get_knowledge_store().search(
            question=question,
            disease_code=disease_code,
            k=top_k,
        )
    except Exception as exc:
        return {
            "success": False,
            "error": type(exc).__name__,
            "message": str(exc),
        }

    return {
        "success": True,
        "rag": True,
        "query": question,
        "disease_code": disease_code,
        "result_count": len(results),
        "results": results,
        "instructions": (
            "回答只能使用检索结果明确提供的信息；"
            "缺少的内容必须说明证据不足。"
        ),
    }


TOOLS: list[BaseTool] = [
    detect_rice_disease,
    search_rice_knowledge,
]

TOOL_MAP: dict[str, BaseTool] = {
    current_tool.name: current_tool
    for current_tool in TOOLS
}
