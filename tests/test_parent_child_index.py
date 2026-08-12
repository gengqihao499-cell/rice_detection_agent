from __future__ import annotations

from langchain_core.documents import Document

from rice_agent.services.rag_store import RiceKnowledgeStore


def _word_count(text: str) -> int:
    return len(text.split())


def test_parent_child_chunk_sizes_and_relationship(monkeypatch) -> None:
    store = RiceKnowledgeStore()
    monkeypatch.setattr(
        store,
        "_token_length_function",
        lambda: (_word_count, "test_words"),
    )
    monkeypatch.setattr(store, "_get_tokenizer", lambda: None)
    text = "\n\n".join(
        " ".join(f"token{section}_{index}" for index in range(100)) + "."
        for section in range(12)
    )
    children = store._split_documents(
        [
            Document(
                page_content=text,
                metadata={
                    "source_id": "unit-test",
                    "disease_code": "leaf_blast",
                },
            )
        ]
    )

    parents = store.load_parents()
    assert len(parents) >= 2
    assert len(children) > len(parents)
    assert max(_word_count(item.page_content) for item in parents.values()) <= 500
    assert max(_word_count(item.page_content) for item in children) <= 150
    assert all(item.metadata["parent_id"] in parents for item in children)

    child = children[0]
    expanded = store.expand_to_parents(
        [
            {
                "content": child.page_content,
                "metadata": child.metadata,
                "relevance_score": 0.91,
            }
        ],
        1,
    )
    assert expanded[0]["metadata"]["chunk_level"] == "parent"
    assert child.metadata["chunk_id"] in expanded[0]["metadata"]["matching_child_ids"]
    assert len(expanded[0]["content"]) > len(child.page_content)


class _CharacterTokenizer:
    name_or_path = "unit-character-tokenizer"

    def __call__(self, text: str, **_: object) -> dict[str, list]:
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def decode(self, ids: list[int], **_: object) -> str:
        return "".join("字" for _ in ids)


def test_token_window_uses_offsets_to_preserve_original_text() -> None:
    store = RiceKnowledgeStore()
    original = "水稻叶瘟病斑呈梭形。" * 80
    children = store._split_documents_with_tokenizer(
        [Document(page_content=original, metadata={"source_id": "offset-test"})],
        _CharacterTokenizer(),
    )
    parents = store.load_parents()
    assert children
    assert parents
    assert next(iter(parents.values())).page_content.startswith("水稻叶瘟")
    assert "水 稻" not in next(iter(parents.values())).page_content
    assert max(item.metadata["token_count"] for item in parents.values()) <= 500
    assert max(item.metadata["token_count"] for item in children) <= 150
