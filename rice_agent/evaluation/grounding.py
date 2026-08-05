from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_CLAIM_SPLIT = re.compile(r"[。！？!?；;\n]+")
_LATIN_TOKEN = re.compile(r"[a-zA-Z0-9_]+")
_CHINESE = re.compile(r"[\u4e00-\u9fff]+")


def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    return str(content).strip()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(cleaned)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("模型评估结果不是 JSON 对象")
    return data


def claims(text: str) -> list[str]:
    result: list[str] = []
    non_factual_prefixes = (
        "参考来源",
        "来源：",
        "路由：",
        "生成答案未通过",
        "当前知识库",
        "当前未配置",
        "请补充",
        "请结合",
        "也可联系",
        "结果仅供",
        "无法给出",
    )
    for raw in _CLAIM_SPLIT.split(text):
        item = raw.strip(" -•\t#*：:")
        item = re.sub(r"[（(][^（）()]*\.md[）)]$", "", item).strip()
        if len(item) < 5:
            continue
        if item.startswith(non_factual_prefixes):
            continue
        result.append(item)
    return result


def lexical_tokens(text: str) -> set[str]:
    lowered = text.lower()
    tokens = set(_LATIN_TOKEN.findall(lowered))
    for segment in _CHINESE.findall(lowered):
        if len(segment) == 1:
            tokens.add(segment)
        else:
            tokens.update(segment[index : index + 2] for index in range(len(segment) - 1))
    return tokens


def support_ratio(statement: str, evidence: str) -> float:
    statement_tokens = lexical_tokens(statement)
    if not statement_tokens:
        return 1.0
    evidence_tokens = lexical_tokens(evidence)
    return len(statement_tokens & evidence_tokens) / len(statement_tokens)


@dataclass(frozen=True, slots=True)
class FaithfulnessResult:
    score: float
    supported_claims: int
    total_claims: int
    unsupported_claims: list[str]
    reason: str
    method: str

    @property
    def passed(self) -> bool:
        return self.score >= 0.9

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FaithfulnessGuard:
    """在答案发送前做声明级忠实度门禁。"""

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = (
            llm.with_config({"tags": ["nostream", "faithfulness-guard"]})
            if llm is not None
            else None
        )

    async def check(
        self,
        answer: str,
        contexts: list[str],
    ) -> FaithfulnessResult:
        if not answer.strip():
            return FaithfulnessResult(0.0, 0, 1, ["空答案"], "答案为空", "rule")
        if not contexts:
            return self._lexical_check(answer, contexts)
        if self.llm is None:
            return self._lexical_check(answer, contexts)

        evidence = "\n\n".join(
            f"[{index}] {context}" for index, context in enumerate(contexts, 1)
        )[:16000]
        prompt = {
            "answer": answer[:10000],
            "retrieved_contexts": evidence,
            "task": (
                "把答案拆成可核查事实声明，逐条判断是否能由检索证据直接支持。"
                "建议、免责声明和明确标注的不确定性不作为事实声明。"
            ),
        }
        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "你是严格的中文 RAG 忠实度审计器。只输出 JSON："
                            '{"supported_claims":整数,"total_claims":整数,'
                            '"unsupported_claims":["..."],"reason":"..."}。'
                            "supported_claims 不得大于 total_claims。"
                        )
                    ),
                    HumanMessage(content=json.dumps(prompt, ensure_ascii=False)),
                ]
            )
            data = parse_json_object(message_text(response))
            total = max(1, int(data.get("total_claims", 1)))
            supported = max(0, min(total, int(data.get("supported_claims", 0))))
            unsupported = [
                str(item)[:300]
                for item in data.get("unsupported_claims", [])
                if str(item).strip()
            ][:8]
            return FaithfulnessResult(
                score=round(supported / total, 4),
                supported_claims=supported,
                total_claims=total,
                unsupported_claims=unsupported,
                reason=str(data.get("reason", "声明级证据核验"))[:500],
                method="llm_claim_audit",
            )
        except Exception as exc:
            fallback = self._lexical_check(answer, contexts)
            return FaithfulnessResult(
                score=fallback.score,
                supported_claims=fallback.supported_claims,
                total_claims=fallback.total_claims,
                unsupported_claims=fallback.unsupported_claims,
                reason=f"LLM 门禁失败，使用词法回退：{type(exc).__name__}",
                method="lexical_fallback",
            )

    def _lexical_check(
        self,
        answer: str,
        contexts: list[str],
    ) -> FaithfulnessResult:
        statements = claims(answer)
        if not statements:
            return FaithfulnessResult(1.0, 1, 1, [], "没有可核查事实声明", "lexical")

        evidence = "\n".join(contexts)
        supported: list[str] = []
        unsupported: list[str] = []
        safe_markers = ("可能", "建议", "请", "证据不足", "无法判断", "仅供", "需结合")

        for statement in statements:
            ratio = support_ratio(statement, evidence)
            is_safe = any(marker in statement for marker in safe_markers)
            if contexts and (ratio >= 0.34 or statement[:12] in evidence):
                supported.append(statement)
            elif is_safe and not contexts:
                supported.append(statement)
            else:
                unsupported.append(statement)

        total = len(statements)
        return FaithfulnessResult(
            score=round(len(supported) / total, 4),
            supported_claims=len(supported),
            total_claims=total,
            unsupported_claims=unsupported[:8],
            reason="基于中文二元词与限定语的轻量回退检查",
            method="lexical",
        )
