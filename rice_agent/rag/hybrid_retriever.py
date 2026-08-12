from __future__ import annotations

import asyncio
import math
import re
from collections import Counter
from typing import Any, Iterable

from rice_agent.config import settings
from rice_agent.rag.query_expansion import QueryPlan


_LATIN = re.compile(r"[a-zA-Z0-9_]+")
_CHINESE = re.compile(r"[\u4e00-\u9fff]+")


def lexical_terms(text: str) -> list[str]:
    lowered = text.lower()
    terms = _LATIN.findall(lowered)
    for segment in _CHINESE.findall(lowered):
        if len(segment) == 1:
            terms.append(segment)
        else:
            terms.extend(
                segment[index : index + 2]
                for index in range(len(segment) - 1)
            )
    return terms


def _candidate_key(item: dict[str, Any]) -> str:
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        if metadata.get("parent_id"):
            return str(metadata["parent_id"])
        if metadata.get("chunk_id"):
            return str(metadata["chunk_id"])
    return " ".join(str(item.get("content", "")).split()).casefold()


def reciprocal_rank_fusion(
    ranked_lists: Iterable[tuple[str, list[dict[str, Any]]]],
    *,
    rrf_k: int = 60,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """RRF(d)=Σ 1/(k+rank)，并按 chunk_id/content 合并去重。"""
    if rrf_k < 1:
        raise ValueError("rrf_k必须大于0")

    fused: dict[str, dict[str, Any]] = {}
    for channel, results in ranked_lists:
        for rank, raw in enumerate(results, 1):
            key = _candidate_key(raw)
            if not key:
                continue
            relevance = float(raw.get("relevance_score", 0.0) or 0.0)
            if key not in fused:
                item = dict(raw)
                item["metadata"] = dict(raw.get("metadata") or {})
                item["rrf_score"] = 0.0
                item["retrieval_channels"] = []
                item["channel_ranks"] = {}
                item["max_retrieval_score"] = relevance
                fused[key] = item
            item = fused[key]
            item["rrf_score"] = float(item["rrf_score"]) + 1.0 / (
                rrf_k + rank
            )
            item["max_retrieval_score"] = max(
                float(item.get("max_retrieval_score", 0.0)),
                relevance,
            )
            if channel not in item["retrieval_channels"]:
                item["retrieval_channels"].append(channel)
            item["channel_ranks"][channel] = rank

    ordered = sorted(
        fused.values(),
        key=lambda item: (
            float(item.get("rrf_score", 0.0)),
            float(item.get("max_retrieval_score", 0.0)),
        ),
        reverse=True,
    )
    if top_n is not None:
        ordered = ordered[:top_n]
    for rank, item in enumerate(ordered, 1):
        item["fusion_rank"] = rank
        item["rrf_score"] = round(float(item["rrf_score"]), 8)
    return ordered


class BM25Retriever:
    def __init__(
        self,
        documents: Iterable[Any],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[dict[str, Any]] = []
        self.term_frequencies: list[Counter[str]] = []
        self.lengths: list[int] = []
        document_frequency: Counter[str] = Counter()

        for document in documents:
            if isinstance(document, dict):
                content = str(document.get("content", ""))
                metadata = dict(document.get("metadata") or {})
            else:
                content = str(getattr(document, "page_content", ""))
                metadata = dict(getattr(document, "metadata", {}) or {})
            terms = lexical_terms(content)
            self.documents.append({"content": content, "metadata": metadata})
            counts = Counter(terms)
            self.term_frequencies.append(counts)
            self.lengths.append(len(terms))
            document_frequency.update(counts.keys())

        self.document_count = len(self.documents)
        self.average_length = (
            sum(self.lengths) / self.document_count
            if self.document_count
            else 0.0
        )
        self.idf = {
            term: math.log(
                1.0
                + (self.document_count - frequency + 0.5)
                / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

    def search(
        self,
        query: str,
        disease_code: str | None,
        k: int,
    ) -> list[dict[str, Any]]:
        query_terms = lexical_terms(query)
        if not query_terms or not self.documents:
            return []

        scores: list[tuple[int, float]] = []
        for index, (document, counts, length) in enumerate(
            zip(self.documents, self.term_frequencies, self.lengths)
        ):
            if disease_code and str(
                document["metadata"].get("disease_code", "")
            ) != disease_code:
                continue
            score = 0.0
            normalization = 1.0 - self.b
            if self.average_length:
                normalization += self.b * length / self.average_length
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                numerator = frequency * (self.k1 + 1.0)
                denominator = frequency + self.k1 * normalization
                score += self.idf.get(term, 0.0) * numerator / denominator
            if score > 0.0:
                scores.append((index, score))

        scores.sort(key=lambda pair: pair[1], reverse=True)
        selected = scores[:k]
        max_score = selected[0][1] if selected else 1.0
        results: list[dict[str, Any]] = []
        for rank, (index, score) in enumerate(selected, 1):
            document = self.documents[index]
            results.append(
                {
                    "rank": rank,
                    "relevance_score": round(score / max_score, 4),
                    "bm25_score": round(score, 6),
                    "content": document["content"],
                    "metadata": {
                        **document["metadata"],
                        "retrieval_backend": "bm25",
                    },
                }
            )
        return results


class BgeReranker:
    """BGE CrossEncoder 精排；加载失败时使用可解释的词法精排。"""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._load_error: str = ""

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                settings.reranker_model_name,
                device=settings.reranker_device,
            )
        return self._model

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            factor = math.exp(-value)
            return 1.0 / (1.0 + factor)
        factor = math.exp(value)
        return factor / (1.0 + factor)

    @staticmethod
    def _lexical_rerank(
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        query_terms = set(lexical_terms(query))
        max_rrf = max(
            (float(item.get("rrf_score", 0.0)) for item in candidates),
            default=1.0,
        ) or 1.0
        ranked: list[dict[str, Any]] = []
        for raw in candidates:
            item = dict(raw)
            content_terms = set(lexical_terms(str(item.get("content", ""))))
            coverage = (
                len(query_terms & content_terms) / len(query_terms)
                if query_terms
                else 0.0
            )
            retrieval_score = min(
                1.0,
                max(0.0, float(item.get("max_retrieval_score", 0.0))),
            )
            normalized_rrf = float(item.get("rrf_score", 0.0)) / max_rrf
            score = 0.55 * coverage + 0.25 * retrieval_score + 0.20 * normalized_rrf
            item["rerank_score"] = round(score, 6)
            item["relevance_score"] = round(score, 4)
            item["rerank_method"] = "lexical_fallback"
            ranked.append(item)
        return sorted(
            ranked,
            key=lambda item: float(item["rerank_score"]),
            reverse=True,
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if not settings.reranker_enabled:
            return self._lexical_rerank(query, candidates)[:top_k]

        try:
            model = self._load()
            from torch import nn

            raw_scores = model.predict(
                [[query, str(item.get("content", ""))] for item in candidates],
                batch_size=settings.reranker_batch_size,
                show_progress_bar=False,
                activation_fn=nn.Identity(),
            )
            ranked: list[dict[str, Any]] = []
            for raw, raw_score in zip(candidates, raw_scores):
                item = dict(raw)
                score = self._sigmoid(float(raw_score))
                item["rerank_raw_score"] = round(float(raw_score), 6)
                item["rerank_score"] = round(score, 6)
                item["relevance_score"] = round(score, 4)
                item["rerank_method"] = settings.reranker_model_name
                ranked.append(item)
            ranked.sort(
                key=lambda item: float(item["rerank_score"]),
                reverse=True,
            )
            return ranked[:top_k]
        except Exception as exc:
            self._load_error = type(exc).__name__
            ranked = self._lexical_rerank(query, candidates)[:top_k]
            for item in ranked:
                item["rerank_fallback_reason"] = self._load_error
            return ranked


class HybridRetriever:
    def __init__(
        self,
        store: Any,
        reranker: BgeReranker | None = None,
    ) -> None:
        self.store = store
        self.bm25 = BM25Retriever(store.load_chunks())
        self.reranker = reranker or BgeReranker()

    def _bm25_search(
        self,
        query: str,
        disease_code: str | None,
        k: int,
    ) -> list[dict[str, Any]]:
        child_results = self.bm25.search(query, disease_code, max(k * 4, k))
        expand = getattr(self.store, "expand_to_parents", None)
        if callable(expand):
            return expand(child_results, k)
        return child_results[:k]

    async def search(
        self,
        question: str,
        disease_code: str | None,
        k: int,
        *,
        query_plan: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        plan = (
            QueryPlan.from_dict(query_plan)
            if query_plan
            else QueryPlan.passthrough(question)
        )
        candidate_k = max(k, settings.rag_candidate_top_k)
        jobs: list[tuple[str, Any]] = []
        for index, query in enumerate(plan.vector_queries):
            jobs.append(
                (
                    f"vector_{index}",
                    asyncio.to_thread(
                        self.store.search,
                        query,
                        disease_code,
                        candidate_k,
                    ),
                )
            )
        for index, query in enumerate(plan.lexical_queries):
            jobs.append(
                (
                    f"bm25_{index}",
                    asyncio.to_thread(
                        self._bm25_search,
                        query,
                        disease_code,
                        candidate_k,
                    ),
                )
            )
        if disease_code:
            # 类别过滤优先保证精度，同时保留两条全库通道用于补足 3–5 个候选，
            # 也避免视觉模型误分类时把正确知识完全挡在召回范围外。
            jobs.extend(
                [
                    (
                        "vector_global",
                        asyncio.to_thread(
                            self.store.search,
                            plan.rewritten_query,
                            None,
                            candidate_k,
                        ),
                    ),
                    (
                        "bm25_global",
                        asyncio.to_thread(
                            self._bm25_search,
                            plan.rewritten_query,
                            None,
                            candidate_k,
                        ),
                    ),
                ]
            )

        gathered = await asyncio.gather(
            *(job for _, job in jobs),
            return_exceptions=True,
        )
        ranked_lists: list[tuple[str, list[dict[str, Any]]]] = []
        errors: list[str] = []
        for (channel, _), result in zip(jobs, gathered):
            if isinstance(result, BaseException):
                errors.append(f"{channel}:{type(result).__name__}")
                continue
            if isinstance(result, list) and result:
                ranked_lists.append((channel, result))

        fused = reciprocal_rank_fusion(
            ranked_lists,
            rrf_k=settings.rrf_k,
            top_n=settings.rag_fusion_top_k,
        )
        final_k = max(3, min(5, k or settings.rag_final_top_k))
        reranked = await asyncio.to_thread(
            self.reranker.rerank,
            plan.rewritten_query,
            fused,
            final_k,
        )
        trace = {
            "query_method": plan.method,
            "hyde_used": bool(plan.hyde_document),
            "vector_query_count": len(plan.vector_queries),
            "bm25_query_count": len(plan.lexical_queries),
            "active_channels": [name for name, _ in ranked_lists],
            "rrf_k": settings.rrf_k,
            "fused_count": len(fused),
            "final_count": len(reranked),
            "errors": errors,
        }
        for rank, item in enumerate(reranked, 1):
            item["rank"] = rank
            item["retrieval_trace"] = trace
        return reranked
