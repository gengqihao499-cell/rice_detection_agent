import pytest

from rice_agent.evaluation.grounding import FaithfulnessResult
from rice_agent.rag.graph_agent import LangGraphRiceRagAgent


CONTEXT = {
    "rank": 1,
    "relevance_score": 0.94,
    "content": "稻瘟病叶片常见梭形病斑，中央灰白，边缘褐色。",
    "metadata": {
        "source": "leaf_blast.md",
        "disease_code": "leaf_blast",
        "chunk_id": "leaf_blast_0001",
    },
}


class SequenceGuard:
    def __init__(self, scores: list[float]) -> None:
        self.scores = iter(scores)

    async def check(self, answer: str, contexts: list[str]) -> FaithfulnessResult:
        score = next(self.scores)
        return FaithfulnessResult(
            score=score,
            supported_claims=1 if score >= 0.9 else 0,
            total_claims=1,
            unsupported_claims=[] if score >= 0.9 else ["无证据声明"],
            reason="test",
            method="test",
        )


async def collect(agent: LangGraphRiceRagAgent) -> list[dict]:
    return [
        event
        async for event in agent.stream(
            question="稻瘟病叶片有什么症状？",
            history=[],
        )
    ]


@pytest.mark.asyncio
async def test_graph_streams_precise_route_and_answer() -> None:
    agent = LangGraphRiceRagAgent(
        retriever=lambda question, disease_code, k: [CONTEXT],
        detector=lambda path: {},
        llm=None,
        guard=SequenceGuard([1.0]),
    )
    events = await collect(agent)
    route = next(event for event in events if event["event"] == "route")
    assert route["data"]["mode"] == "precise_hit"
    assert any(event["event"] == "answer_delta" for event in events)
    assert events[-1]["event"] == "graph_complete"


@pytest.mark.asyncio
async def test_graph_retries_after_failed_guard() -> None:
    agent = LangGraphRiceRagAgent(
        retriever=lambda question, disease_code, k: [CONTEXT],
        detector=lambda path: {},
        llm=None,
        guard=SequenceGuard([0.5, 1.0]),
    )
    events = await collect(agent)
    guard_states = [event["state"] for event in events if event["event"] == "guard"]
    assert guard_states == ["failed", "passed"]
    assert events[-1]["retry_count"] == 1
