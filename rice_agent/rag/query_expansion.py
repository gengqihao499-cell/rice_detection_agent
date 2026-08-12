from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from rice_agent.config import settings
from rice_agent.evaluation.grounding import message_text, parse_json_object


def _unique_texts(values: list[str], limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = " ".join(str(raw).split()).strip()
        key = value.casefold()
        if not value or key in seen:
            continue
        seen.add(key)
        result.append(value[:500])
        if limit is not None and len(result) >= limit:
            break
    return result


def _fallback_queries(question: str, count: int) -> list[str]:
    suffixes = ["症状与鉴别", "发病原因与传播条件", "防治与田间管理"]
    return [f"{question} {suffix}" for suffix in suffixes[:count]]


@dataclass(frozen=True, slots=True)
class QueryPlan:
    original_query: str
    rewritten_query: str
    hyde_document: str
    multi_queries: list[str]
    method: str

    @classmethod
    def passthrough(cls, question: str) -> "QueryPlan":
        cleaned = " ".join(question.split()).strip()
        return cls(
            original_query=cleaned,
            rewritten_query=cleaned,
            hyde_document="",
            multi_queries=_fallback_queries(cleaned, settings.multi_query_count),
            method="deterministic_fallback",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryPlan":
        original = str(data.get("original_query", "")).strip()
        fallback = cls.passthrough(original)
        multi = data.get("multi_queries", fallback.multi_queries)
        return cls(
            original_query=original,
            rewritten_query=str(data.get("rewritten_query") or original).strip(),
            hyde_document=str(data.get("hyde_document") or "").strip(),
            multi_queries=_unique_texts(
                list(multi) if isinstance(multi, list) else fallback.multi_queries,
                settings.multi_query_count,
            ),
            method=str(data.get("method") or fallback.method),
        )

    @property
    def vector_queries(self) -> list[str]:
        values = [self.original_query, self.rewritten_query]
        if settings.hyde_enabled and self.hyde_document:
            values.append(self.hyde_document)
        values.extend(self.multi_queries)
        return _unique_texts(values)

    @property
    def lexical_queries(self) -> list[str]:
        return _unique_texts(
            [self.original_query, self.rewritten_query, *self.multi_queries]
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["vector_query_count"] = len(self.vector_queries)
        result["lexical_query_count"] = len(self.lexical_queries)
        result["hyde_used"] = bool(self.hyde_document)
        return result


class QueryEnhancer:
    """一次 LLM 调用完成独立问题改写、HyDE 与多 Query 扩展。"""

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = (
            llm.with_config({"tags": ["nostream", "query-enhancement"]})
            if llm is not None
            else None
        )

    async def enhance(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> QueryPlan:
        fallback = QueryPlan.passthrough(question)
        if not settings.query_enhancement_enabled or self.llm is None:
            return fallback

        history_text = [
            {
                "role": str(item.get("role", "user")),
                "content": str(item.get("content", ""))[:1200],
            }
            for item in (history or [])[-4:]
        ]
        messages = [
            SystemMessage(
                content=(
                    "你是农业检索查询规划器，只返回 JSON。"
                    "rewritten_query 要把当前问题改写成无需对话上下文也能理解的检索句；"
                    "hyde_document 要写一段约 80-160 字的假设性知识库答案，仅用于向量检索，"
                    "不能作为最终诊断；multi_queries 从症状鉴别、病因条件、防治管理等不同角度"
                    f"生成 {settings.multi_query_count} 条短查询。不要输出 Markdown。"
                )
            ),
            HumanMessage(
                content=json.dumps(
                    {
                        "question": fallback.original_query,
                        "recent_history": history_text,
                        "schema": {
                            "rewritten_query": "string",
                            "hyde_document": "string",
                            "multi_queries": ["string"],
                        },
                    },
                    ensure_ascii=False,
                )
            ),
        ]
        try:
            response = await self.llm.ainvoke(messages)
            data = parse_json_object(message_text(response))
        except Exception:
            return fallback

        rewritten = " ".join(
            str(data.get("rewritten_query") or fallback.original_query).split()
        ).strip()[:500]
        hyde = " ".join(str(data.get("hyde_document") or "").split()).strip()
        raw_multi = data.get("multi_queries", [])
        multi = _unique_texts(
            list(raw_multi) if isinstance(raw_multi, list) else [],
            settings.multi_query_count,
        )
        if len(multi) < settings.multi_query_count:
            multi = _unique_texts(
                [*multi, *_fallback_queries(rewritten, settings.multi_query_count)],
                settings.multi_query_count,
            )
        return QueryPlan(
            original_query=fallback.original_query,
            rewritten_query=rewritten or fallback.original_query,
            hyde_document=hyde[:1200] if settings.hyde_enabled else "",
            multi_queries=multi,
            method="llm_hyde_multi_query",
        )
