from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class RouteMode(StrEnum):
    PRECISE_HIT = "precise_hit"
    REFERENCE_GENERATION = "reference_generation"
    AI_INFERENCE = "ai_inference"


@dataclass(frozen=True, slots=True)
class RouteDecision:
    mode: RouteMode
    label: str
    reason: str
    top_score: float
    qualified_chunks: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload


ROUTE_LABELS = {
    RouteMode.PRECISE_HIT: "精准命中",
    RouteMode.REFERENCE_GENERATION: "参考生成",
    RouteMode.AI_INFERENCE: "AI 推断",
}


def _score(item: dict[str, Any]) -> float:
    try:
        return max(0.0, min(1.0, float(item.get("relevance_score", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def decide_route(
    contexts: list[dict[str, Any]],
    *,
    precise_threshold: float,
    reference_threshold: float,
    min_precise_chunks: int = 1,
) -> RouteDecision:
    """根据 Chroma 相关度选择三级 RAG 策略。"""
    if not 0.0 <= reference_threshold <= precise_threshold <= 1.0:
        raise ValueError("路由阈值必须满足 0 <= 参考阈值 <= 精准阈值 <= 1")
    if min_precise_chunks < 1:
        raise ValueError("min_precise_chunks 必须大于等于 1")

    scores = sorted((_score(item) for item in contexts), reverse=True)
    top_score = scores[0] if scores else 0.0
    qualified = sum(score >= precise_threshold for score in scores)

    if qualified >= min_precise_chunks:
        mode = RouteMode.PRECISE_HIT
        reason = f"{qualified} 个知识块达到精准阈值 {precise_threshold:.2f}"
    elif top_score >= reference_threshold:
        mode = RouteMode.REFERENCE_GENERATION
        reason = (
            f"最高相关度 {top_score:.2f} 未达到精准阈值，"
            "采用有边界的参考生成"
        )
    else:
        mode = RouteMode.AI_INFERENCE
        reason = (
            f"最高相关度 {top_score:.2f} 低于参考阈值 "
            f"{reference_threshold:.2f}"
        )

    return RouteDecision(
        mode=mode,
        label=ROUTE_LABELS[mode],
        reason=reason,
        top_score=round(top_score, 4),
        qualified_chunks=qualified,
    )
