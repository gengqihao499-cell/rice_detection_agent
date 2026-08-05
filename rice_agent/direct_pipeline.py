from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

from rice_agent.agent.tools import (
    detect_rice_disease,
    search_rice_knowledge,
)
from rice_agent.config import settings


DIRECT_SYSTEM_PROMPT = """
你是水稻病虫害辅助分析助手。
只能依据给定的YOLO检测结果和RAG证据回答。
不得编造药剂名称、剂量或当地法规。
输出应包含检测结果、置信度、证据、管理建议、来源与局限。
"""


def analyze_image_direct(
    image_path: str,
    question: str = "请分析可能的病虫害并给出基础管理建议。",
) -> dict[str, Any]:
    """
    确定性编排备选方案：
    YOLO -> 按类别检索RAG -> LLM总结。
    当自定义DeepSeek网关不支持tool_calls时可使用。
    """
    detection = detect_rice_disease.invoke(
        {
            "image_path": image_path,
            "confidence_threshold": settings.yolo_confidence,
            "iou_threshold": settings.yolo_iou,
        }
    )

    if not detection.get("success"):
        return {
            "success": False,
            "detection": detection,
            "answer": detection.get("message", "检测失败"),
        }

    summaries = detection.get("class_summary", [])
    abnormal = [
        item
        for item in summaries
        if item.get("kind") in {"disease", "pest"}
    ]
    candidates = abnormal[:3] or summaries[:1]

    evidence: list[dict[str, Any]] = []

    for candidate in candidates:
        code = candidate["disease_code"]
        retrieved = search_rice_knowledge.invoke(
            {
                "question": question,
                "disease_code": code,
                "top_k": settings.rag_top_k,
            }
        )
        evidence.append(
            {
                "candidate": candidate,
                "retrieval": retrieved,
            }
        )

    if not settings.deepseek_api_key:
        return {
            "success": True,
            "detection": detection,
            "evidence": evidence,
            "answer": (
                "未配置DEEPSEEK_API_KEY，已完成YOLO检测与RAG检索，"
                "请查看detection和evidence字段。"
            ),
        }

    llm = ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
        max_retries=2,
    )

    user_payload = {
        "question": question,
        "detection": detection,
        "rag_evidence": evidence,
    }

    response = llm.invoke(
        [
            SystemMessage(content=DIRECT_SYSTEM_PROMPT.strip()),
            HumanMessage(
                content=json.dumps(
                    user_payload,
                    ensure_ascii=False,
                    default=str,
                )
            ),
        ]
    )

    return {
        "success": True,
        "detection": detection,
        "evidence": evidence,
        "answer": str(response.content),
    }
