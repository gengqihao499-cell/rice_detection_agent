from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from rice_agent.config import PROJECT_ROOT, settings
from rice_agent.services.rag_store import RiceKnowledgeStore


DEFAULT_MODELS = list(settings.embedding_benchmark_models)
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
QWEN3_QUERY_INSTRUCTION = (
    "Instruct: Given a query about rice diseases, pests, symptoms, diagnosis, "
    "and crop management, retrieve relevant passages that answer the query\n"
    "Query: "
)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[max(0, index)]


def _query_text(model_name: str, query: str) -> str:
    lowered = model_name.casefold()
    if "qwen3-embedding" in lowered:
        return QWEN3_QUERY_INSTRUCTION + query
    if "bge-" in lowered and "bge-m3" not in lowered:
        return BGE_QUERY_INSTRUCTION + query
    return query


def _evaluate_model(
    model_name: str,
    device: str,
    corpus: list[str],
    disease_codes: list[str],
    samples: list[dict[str, Any]],
    *,
    local_files_only: bool,
    trust_remote_code: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = SentenceTransformer(
        model_name,
        device=device,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    load_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    corpus_vectors = model.encode(
        corpus,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    corpus_encode_ms = (time.perf_counter() - started) * 1000

    ranks: list[int | None] = []
    latencies: list[float] = []
    examples: list[dict[str, Any]] = []
    for sample in samples:
        query = str(sample["query"])
        relevant = set(sample["relevant_disease_codes"])
        started = time.perf_counter()
        query_vector = model.encode(
            [_query_text(model_name, query)],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        similarities = np.asarray(corpus_vectors) @ np.asarray(query_vector)
        order = np.argsort(-similarities)
        latencies.append((time.perf_counter() - started) * 1000)
        first_rank: int | None = None
        for rank, index in enumerate(order, 1):
            if disease_codes[int(index)] in relevant:
                first_rank = rank
                break
        ranks.append(first_rank)
        examples.append(
            {
                "query": query,
                "first_relevant_rank": first_rank,
                "top1_disease_code": disease_codes[int(order[0])],
            }
        )

    count = len(samples)
    metrics: dict[str, float] = {}
    for cutoff in (1, 3, 5):
        metrics[f"hit@{cutoff}"] = round(
            sum(rank is not None and rank <= cutoff for rank in ranks) / count,
            4,
        )
    metrics["mrr@5"] = round(
        sum(1.0 / rank for rank in ranks if rank is not None and rank <= 5)
        / count,
        4,
    )
    return {
        "model": model_name,
        "embedding_dimension": int(np.asarray(corpus_vectors).shape[1]),
        "query_count": count,
        **metrics,
        "load_ms": round(load_ms, 2),
        "corpus_encode_ms": round(corpus_encode_ms, 2),
        "mean_query_ms": round(statistics.mean(latencies), 2),
        "p95_query_ms": round(_percentile(latencies, 0.95), 2),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="在项目标注集上比较中文嵌入模型的 Hit@K、MRR 与延迟。"
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="只测试本地已下载模型，不访问 Hugging Face。",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="模型明确要求时才开启；Qwen3 SentenceTransformers 通常不需要。",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "retrieval_queries.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "embedding_benchmark.json",
    )
    args = parser.parse_args()

    samples = json.loads(args.dataset.read_text(encoding="utf-8"))
    chunks = [
        document
        for document in RiceKnowledgeStore().load_chunks()
        if str(document.metadata.get("disease_code") or "")
    ]
    corpus = [document.page_content for document in chunks]
    disease_codes = [str(document.metadata["disease_code"]) for document in chunks]

    results: list[dict[str, Any]] = []
    for model_name in args.models:
        try:
            results.append(
                _evaluate_model(
                    model_name,
                    args.device,
                    corpus,
                    disease_codes,
                    samples,
                    local_files_only=args.local_files_only,
                    trust_remote_code=args.trust_remote_code,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "model": model_name,
                    "error": type(exc).__name__,
                    "message": str(exc)[:500],
                }
            )

    try:
        dataset_label = str(args.dataset.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        dataset_label = str(args.dataset)
    report = {
        "dataset": dataset_label,
        "corpus_chunks": len(corpus),
        "query_count": len(samples),
        "device": args.device,
        "local_files_only": args.local_files_only,
        "metrics": {
            "hit@k": "前k个知识块至少有1个属于相关疾病文档的查询占比",
            "mrr@5": "首个相关知识块排名倒数在前5名内的平均值",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
