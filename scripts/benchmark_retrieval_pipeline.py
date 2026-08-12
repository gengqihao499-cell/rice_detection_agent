from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from rice_agent.config import PROJECT_ROOT
from rice_agent.rag.hybrid_retriever import BM25Retriever, HybridRetriever
from rice_agent.rag.query_expansion import QueryPlan
from rice_agent.services.rag_store import RiceKnowledgeStore


SearchFunction = Callable[[str, int], Awaitable[list[dict[str, Any]]]]


def _first_relevant_rank(
    results: list[dict[str, Any]],
    relevant: set[str],
) -> int | None:
    for rank, item in enumerate(results, 1):
        metadata = item.get("metadata") or {}
        if str(metadata.get("disease_code", "")) in relevant:
            return rank
    return None


async def _evaluate(
    name: str,
    samples: list[dict[str, Any]],
    search: SearchFunction,
) -> dict[str, Any]:
    ranks: list[int | None] = []
    latencies: list[float] = []
    for sample in samples:
        started = time.perf_counter()
        results = await search(str(sample["query"]), 5)
        latencies.append((time.perf_counter() - started) * 1000)
        ranks.append(
            _first_relevant_rank(
                results,
                set(sample["relevant_disease_codes"]),
            )
        )

    count = len(samples)
    result: dict[str, Any] = {"pipeline": name, "query_count": count}
    for cutoff in (1, 3, 5):
        result[f"hit@{cutoff}"] = round(
            sum(rank is not None and rank <= cutoff for rank in ranks) / count,
            4,
        )
    result["mrr@5"] = round(
        sum(1.0 / rank for rank in ranks if rank is not None and rank <= 5)
        / count,
        4,
    )
    result["mean_latency_ms"] = round(statistics.mean(latencies), 2)
    return result


async def main_async(args: argparse.Namespace) -> None:
    samples = json.loads(args.dataset.read_text(encoding="utf-8"))
    store = RiceKnowledgeStore()
    bm25 = BM25Retriever(store.load_chunks())
    hybrid = HybridRetriever(store)

    async def vector_search(query: str, k: int) -> list[dict[str, Any]]:
        return await asyncio.to_thread(store.search, query, None, k)

    async def bm25_search(query: str, k: int) -> list[dict[str, Any]]:
        return await asyncio.to_thread(bm25.search, query, None, k)

    async def hybrid_search(query: str, k: int) -> list[dict[str, Any]]:
        plan = QueryPlan.passthrough(query)
        return await hybrid.search(
            query,
            None,
            k,
            query_plan=plan.to_dict(),
        )

    try:
        dataset_label = str(args.dataset.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        dataset_label = str(args.dataset)
    report = {
        "dataset": dataset_label,
        "query_count": len(samples),
        "notes": (
            "离线消融使用确定性多Query扩展与词法ReRank；线上配置DeepSeek后会增加HyDE，"
            "配置BGE reranker后会替换词法精排。"
        ),
        "results": [
            await _evaluate("vector_bge_small", samples, vector_search),
            await _evaluate("bm25", samples, bm25_search),
            await _evaluate("multi_query_rrf_lexical_rerank", samples, hybrid_search),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="检索链路离线消融评测")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "retrieval_queries.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "retrieval_pipeline_benchmark.json",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
