from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rice_agent.rag.hybrid_retriever import (
    BM25Retriever,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from rice_agent.rag.query_expansion import QueryPlan


def _item(chunk_id: str, score: float, content: str = "文本") -> dict:
    return {
        "relevance_score": score,
        "content": content,
        "metadata": {"chunk_id": chunk_id, "disease_code": chunk_id.split("_")[0]},
    }


def test_rrf_uses_formula_and_deduplicates() -> None:
    fused = reciprocal_rank_fusion(
        [
            ("vector", [_item("a", 0.9), _item("b", 0.8)]),
            ("bm25", [_item("b", 1.0), _item("a", 0.7)]),
        ],
        rrf_k=60,
    )
    assert len(fused) == 2
    by_id = {item["metadata"]["chunk_id"]: item for item in fused}
    expected = 1 / 61 + 1 / 62
    assert by_id["a"]["rrf_score"] == pytest.approx(expected, abs=1e-8)
    assert set(by_id["a"]["retrieval_channels"]) == {"vector", "bm25"}


def test_rrf_deduplicates_multiple_children_of_same_parent() -> None:
    first = _item("parent_c000", 0.9)
    second = _item("parent_c001", 0.8)
    first["metadata"]["parent_id"] = "parent"
    second["metadata"]["parent_id"] = "parent"
    fused = reciprocal_rank_fusion(
        [("vector", [first]), ("bm25", [second])],
        rrf_k=60,
    )
    assert len(fused) == 1
    assert fused[0]["rrf_score"] == pytest.approx(2 / 61, abs=1e-8)


def test_bm25_returns_relevant_rice_document() -> None:
    documents = [
        Document(
            page_content="白叶枯病从叶尖叶缘形成水浸状黄橙色条斑",
            metadata={"chunk_id": "blb_1", "disease_code": "bacterial_leaf_blight"},
        ),
        Document(
            page_content="穗颈瘟会造成白穗和穗颈黑褐色病斑",
            metadata={"chunk_id": "neck_1", "disease_code": "neck_blast"},
        ),
    ]
    result = BM25Retriever(documents).search("叶缘水浸状条斑", None, 2)
    assert result[0]["metadata"]["disease_code"] == "bacterial_leaf_blight"
    assert result[0]["relevance_score"] == 1.0


class FakeStore:
    def __init__(self) -> None:
        self.documents = [
            Document(
                page_content="叶瘟病斑呈梭形，中间灰白，边缘红褐。",
                metadata={"chunk_id": "leaf_blast_1", "disease_code": "leaf_blast"},
            ),
            Document(
                page_content="胡麻斑通常呈圆形褐斑。",
                metadata={"chunk_id": "brown_spot_1", "disease_code": "brown_spot"},
            ),
            Document(
                page_content="穗颈瘟可能形成白穗。",
                metadata={"chunk_id": "neck_blast_1", "disease_code": "neck_blast"},
            ),
        ]

    def load_chunks(self) -> list[Document]:
        return self.documents

    def search(self, query: str, disease_code: str | None, k: int) -> list[dict]:
        selected = self.documents
        if disease_code:
            selected = [
                item for item in selected if item.metadata["disease_code"] == disease_code
            ]
        return [
            {
                "rank": rank,
                "relevance_score": 0.9 - rank * 0.1,
                "content": item.page_content,
                "metadata": item.metadata,
            }
            for rank, item in enumerate(selected[:k], 1)
        ]


class FakeReranker:
    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        ranked = []
        for index, raw in enumerate(candidates[:top_k]):
            item = dict(raw)
            item["rerank_score"] = 0.95 - index * 0.1
            item["relevance_score"] = item["rerank_score"]
            item["rerank_method"] = "test"
            ranked.append(item)
        return ranked


@pytest.mark.asyncio
async def test_hybrid_retriever_returns_final_three_to_five() -> None:
    retriever = HybridRetriever(FakeStore(), reranker=FakeReranker())
    plan = QueryPlan.passthrough("梭形灰白病斑是什么")
    result = await retriever.search(
        plan.original_query,
        None,
        4,
        query_plan=plan.to_dict(),
    )
    assert 3 <= len(result) <= 5
    assert result[0]["retrieval_channels"]
    assert result[0]["retrieval_trace"]["rrf_k"] == 60
    assert result[0]["rerank_method"] == "test"
