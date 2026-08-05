from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from rice_agent.evaluation.grounding import (
    claims,
    lexical_tokens,
    message_text,
    parse_json_object,
    support_ratio,
)


def _clamp(value: Any) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True, slots=True)
class EvaluationInput:
    session_id: str
    turn_id: str
    question: str
    answer: str
    contexts: list[str]
    reference_answer: str | None = None
    route_mode: str = ""


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    session_id: str
    turn_id: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    faithfulness_target: float
    target_met: bool
    reference_mode: str
    method: str
    reasons: dict[str, str] = field(default_factory=dict)
    latency_ms: float = 0.0
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RagasLightEvaluator:
    """一次异步判分完成四项 RAGAS 核心指标，避免四次串行 LLM 调用。"""

    def __init__(self, llm: Any | None = None, faithfulness_target: float = 0.9) -> None:
        self.llm = (
            llm.with_config({"tags": ["nostream", "ragas-light"]})
            if llm is not None
            else None
        )
        self.faithfulness_target = faithfulness_target

    async def evaluate(self, sample: EvaluationInput) -> EvaluationResult:
        started = time.perf_counter()
        reference_mode = "gold_reference" if sample.reference_answer else "online_proxy"
        if self.llm is not None:
            try:
                result = await self._llm_evaluate(sample, reference_mode)
                return EvaluationResult(
                    **result,
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                )
            except Exception:
                pass

        result = self._lexical_evaluate(sample, reference_mode)
        return EvaluationResult(
            **result,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    async def _llm_evaluate(
        self,
        sample: EvaluationInput,
        reference_mode: str,
    ) -> dict[str, Any]:
        payload = {
            "question": sample.question[:4000],
            "answer": sample.answer[:10000],
            "retrieved_contexts": sample.contexts[:8],
            "reference_answer": sample.reference_answer,
            "reference_mode": reference_mode,
        }
        response = await self.llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        "你是 RAGAS-light 中文评估器。一次完成四项 0~1 判分。"
                        "忠实度=答案中可被检索上下文支持的事实声明占比；"
                        "相关性=答案对用户问题的直接回应程度；"
                        "精确率=按检索顺序，相关知识块靠前且噪声少的程度；"
                        "召回率=参考答案事实被上下文覆盖的比例。"
                        "若 reference_mode=online_proxy，则用问题所需信息与答案事实作为在线代理，"
                        "并在召回率理由中明确这是代理指标。"
                        "只输出 JSON，键为 faithfulness、answer_relevancy、context_precision、"
                        "context_recall、reasons；reasons 对应四项简短中文理由。"
                    )
                ),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ]
        )
        data = parse_json_object(message_text(response))
        faithfulness = _clamp(data.get("faithfulness"))
        reasons = data.get("reasons") if isinstance(data.get("reasons"), dict) else {}
        return {
            "session_id": sample.session_id,
            "turn_id": sample.turn_id,
            "faithfulness": faithfulness,
            "answer_relevancy": _clamp(data.get("answer_relevancy")),
            "context_precision": _clamp(data.get("context_precision")),
            "context_recall": _clamp(data.get("context_recall")),
            "faithfulness_target": self.faithfulness_target,
            "target_met": faithfulness >= self.faithfulness_target,
            "reference_mode": reference_mode,
            "method": "single_call_llm_judge",
            "reasons": {str(key): str(value)[:500] for key, value in reasons.items()},
        }

    def _lexical_evaluate(
        self,
        sample: EvaluationInput,
        reference_mode: str,
    ) -> dict[str, Any]:
        evidence = "\n".join(sample.contexts)
        answer_claims = claims(sample.answer)
        supported = sum(support_ratio(item, evidence) >= 0.34 for item in answer_claims)
        faithfulness = supported / len(answer_claims) if answer_claims else 1.0

        question_tokens = lexical_tokens(sample.question)
        answer_tokens = lexical_tokens(sample.answer)
        answer_relevancy = (
            len(question_tokens & answer_tokens) / len(question_tokens)
            if question_tokens
            else 0.0
        )

        useful_contexts = 0
        target_text = f"{sample.question}\n{sample.reference_answer or ''}"
        for context in sample.contexts:
            if support_ratio(context[:500], target_text) >= 0.12:
                useful_contexts += 1
        context_precision = useful_contexts / len(sample.contexts) if sample.contexts else 0.0

        reference = sample.reference_answer or sample.question
        reference_claims = claims(reference) or [reference]
        recalled = sum(support_ratio(item, evidence) >= 0.28 for item in reference_claims)
        context_recall = recalled / len(reference_claims) if evidence else 0.0

        faithfulness = _clamp(faithfulness)
        return {
            "session_id": sample.session_id,
            "turn_id": sample.turn_id,
            "faithfulness": faithfulness,
            "answer_relevancy": _clamp(answer_relevancy),
            "context_precision": _clamp(context_precision),
            "context_recall": _clamp(context_recall),
            "faithfulness_target": self.faithfulness_target,
            "target_met": faithfulness >= self.faithfulness_target,
            "reference_mode": reference_mode,
            "method": "lexical_fallback",
            "reasons": {
                "faithfulness": "声明与检索上下文的中文词项覆盖率",
                "answer_relevancy": "问题词项在答案中的覆盖率",
                "context_precision": "相关检索块占全部检索块的比例",
                "context_recall": (
                    "无金标准答案，使用问题信息需求做代理召回率"
                    if reference_mode == "online_proxy"
                    else "参考答案声明的上下文覆盖率"
                ),
            },
        }
