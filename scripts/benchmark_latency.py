from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from rice_agent.evaluation.grounding import FaithfulnessResult
from rice_agent.rag.graph_agent import LangGraphRiceRagAgent


DETECTION_DELAY = 0.04
RETRIEVAL_DELAY = 0.08
GENERATION_DELAY = 0.12
GUARD_DELAY = 0.03
EVALUATION_DELAY = 0.10


def fake_detection(_: str) -> dict[str, Any]:
    time.sleep(DETECTION_DELAY)
    return {
        "success": True,
        "class_summary": [
            {"disease_code": "leaf_blast", "kind": "disease"},
            {"disease_code": "brown_spot", "kind": "disease"},
            {"disease_code": "bacterial_leaf_blight", "kind": "disease"},
        ],
    }


def fake_retrieval(question: str, disease_code: str | None, k: int) -> list[dict[str, Any]]:
    time.sleep(RETRIEVAL_DELAY)
    code = disease_code or "global"
    return [
        {
            "relevance_score": 0.91,
            "content": f"{code} 的受控基准知识证据。",
            "metadata": {
                "source": f"{code}.md",
                "disease_code": code,
                "chunk_id": f"{code}_0001",
            },
        }
    ][:k]


class DelayedLlm:
    def with_config(self, _: dict[str, Any]) -> "DelayedLlm":
        return self

    async def ainvoke(self, _: list[Any]) -> AIMessage:
        await asyncio.sleep(GENERATION_DELAY)
        return AIMessage(
            content=(
                "### 判断\n受控基准答案。\n\n### 依据\n依据检索证据 [1]。\n\n"
                "### 建议\n请结合田间情况复核。\n\n结果仅供辅助判断，不替代专业诊断。"
            )
        )


class DelayedGuard:
    async def check(self, answer: str, contexts: list[str]) -> FaithfulnessResult:
        await asyncio.sleep(GUARD_DELAY)
        return FaithfulnessResult(
            score=1.0,
            supported_claims=1,
            total_claims=1,
            unsupported_claims=[],
            reason="controlled benchmark",
            method="controlled",
        )


async def legacy_once() -> float:
    started = time.perf_counter()
    detection = await asyncio.to_thread(fake_detection, "benchmark.jpg")
    for item in detection["class_summary"][:3]:
        await asyncio.to_thread(
            fake_retrieval,
            "受控基准问题",
            item["disease_code"],
            4,
        )
    await asyncio.sleep(GENERATION_DELAY)
    return (time.perf_counter() - started) * 1000


async def langgraph_once() -> tuple[float, float]:
    agent = LangGraphRiceRagAgent(
        retriever=fake_retrieval,
        detector=fake_detection,
        llm=DelayedLlm(),
        guard=DelayedGuard(),
    )
    started = time.perf_counter()
    answer_ms = 0.0
    async for event in agent.stream(
        question="受控基准问题",
        history=[],
        image_path="benchmark.jpg",
    ):
        if event["event"] == "answer_end":
            answer_ms = (time.perf_counter() - started) * 1000

    await asyncio.sleep(EVALUATION_DELAY)
    quality_ms = (time.perf_counter() - started) * 1000
    return answer_ms, quality_ms


async def benchmark(trials: int) -> dict[str, Any]:
    legacy: list[float] = []
    graph_answer: list[float] = []
    graph_quality: list[float] = []
    for _ in range(trials):
        legacy.append(await legacy_once())
        answer_ms, quality_ms = await langgraph_once()
        graph_answer.append(answer_ms)
        graph_quality.append(quality_ms)

    legacy_median = statistics.median(legacy)
    graph_median = statistics.median(graph_answer)
    improvement = (legacy_median - graph_median) / legacy_median * 100
    return {
        "benchmark_type": "controlled_orchestration",
        "trials": trials,
        "assumptions_ms": {
            "detection": DETECTION_DELAY * 1000,
            "retrieval_per_candidate": RETRIEVAL_DELAY * 1000,
            "generation": GENERATION_DELAY * 1000,
            "faithfulness_guard": GUARD_DELAY * 1000,
            "async_evaluation": EVALUATION_DELAY * 1000,
        },
        "legacy_sequential_response_ms_median": round(legacy_median, 2),
        "langgraph_answer_ms_median": round(graph_median, 2),
        "langgraph_quality_event_ms_median": round(
            statistics.median(graph_quality), 2
        ),
        "answer_latency_improvement_percent": round(improvement, 2),
        "notes": (
            "受控编排基准用于比较串行三级检索与并行检索、异步评估的结构收益；"
            "不代表真实 BGE、YOLO 或 DeepSeek 网络耗时。"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="前后编排响应时间受控基准")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials < 1:
        raise SystemExit("--trials 必须大于 0")
    result = asyncio.run(benchmark(args.trials))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
