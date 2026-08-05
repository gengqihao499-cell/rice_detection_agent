from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from rice_agent.config import settings
from rice_agent.evaluation.grounding import lexical_tokens


@lru_cache(maxsize=1)
def _documents() -> tuple[tuple[Path, str], ...]:
    return tuple(
        (path, path.read_text(encoding="utf-8"))
        for path in sorted(settings.knowledge_dir.glob("*.md"))
    )


def keyword_search(
    question: str,
    disease_code: str | None,
    k: int,
) -> list[dict[str, Any]]:
    """仅在 BGE/Chroma 依赖不可用时启用的可解释降级检索。"""
    query_tokens = lexical_tokens(question)
    results: list[dict[str, Any]] = []
    for path, content in _documents():
        if disease_code and path.stem != disease_code:
            continue
        chunks = [
            item.strip()
            for item in re.split(r"\n(?=##\s)", content)
            if item.strip()
        ]
        for index, chunk in enumerate(chunks):
            chunk_tokens = lexical_tokens(chunk)
            overlap = len(query_tokens & chunk_tokens)
            score = overlap / max(1, len(query_tokens))
            heading = chunk.splitlines()[0].lstrip("# ").strip()
            normalized_heading = heading.replace("水稻", "").split("（", 1)[0]
            if (
                index == 0
                and len(normalized_heading) >= 2
                and normalized_heading in question
            ):
                score = min(1.0, score + 0.52)
            if disease_code and path.stem == disease_code:
                score = min(1.0, score + 0.38)
            results.append(
                {
                    "rank": 0,
                    "relevance_score": round(score, 4),
                    "content": chunk,
                    "metadata": {
                        "source": path.name,
                        "disease_code": path.stem,
                        "chunk_id": f"fallback_{path.stem}_{index:04d}",
                        "retrieval_backend": "keyword_fallback",
                    },
                }
            )

    results.sort(key=lambda item: item["relevance_score"], reverse=True)
    selected = results[:k]
    for rank, item in enumerate(selected, 1):
        item["rank"] = rank
    return selected
