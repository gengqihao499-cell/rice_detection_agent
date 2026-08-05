from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rice_agent.config import settings
from rice_agent.domain import CODE_METADATA


COLLECTION_NAME = "rice_disease_knowledge"


class RiceKnowledgeStore:
    """本地 Markdown + BGE Embedding + Chroma 持久化检索。"""

    def __init__(self) -> None:
        self.knowledge_dir = settings.knowledge_dir
        self.chroma_dir = settings.chroma_dir
        self.manifest_path = self.chroma_dir / "manifest.json"
        self._embeddings: HuggingFaceEmbeddings | None = None
        self._vector_store: Chroma | None = None
        self._lock = Lock()

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model_name,
                model_kwargs={
                    "device": settings.embedding_device,
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                },
            )
        return self._embeddings

    def _markdown_files(self) -> list[Path]:
        files = sorted(self.knowledge_dir.glob("*.md"))

        if not files:
            raise FileNotFoundError(
                f"知识目录中没有Markdown文档：{self.knowledge_dir}"
            )

        return files

    def _digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(settings.embedding_model_name.encode("utf-8"))
        digest.update(str(settings.rag_chunk_size).encode("utf-8"))
        digest.update(str(settings.rag_chunk_overlap).encode("utf-8"))

        for path in self._markdown_files():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())

        return digest.hexdigest()

    def _load_documents(self) -> list[Document]:
        documents: list[Document] = []

        for path in self._markdown_files():
            content = path.read_text(encoding="utf-8").strip()

            if not content:
                continue

            code = path.stem
            domain = CODE_METADATA.get(code, {})

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "disease_code": code,
                        "display_name_zh": domain.get(
                            "display_name_zh",
                            code,
                        ),
                        "kind": domain.get("kind", "unknown"),
                        "source": path.name,
                        "source_path": str(path.resolve()),
                    },
                )
            )

        if not documents:
            raise RuntimeError("知识文档均为空")

        return documents

    def _split_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            separators=[
                "\n## ",
                "\n### ",
                "\n\n",
                "\n",
                "。",
                "！",
                "？",
                "；",
                "，",
                "、",
                " ",
                "",
            ],
            keep_separator=True,
            add_start_index=True,
            length_function=len,
        )
        chunks = splitter.split_documents(documents)

        counters: dict[str, int] = {}

        for chunk in chunks:
            code = str(chunk.metadata["disease_code"])
            index = counters.get(code, 0)
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_id"] = f"{code}_{index:04d}"
            counters[code] = index + 1

        return chunks

    def _manifest_digest(self) -> str | None:
        if not self.manifest_path.is_file():
            return None

        try:
            data = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return None

        return data.get("digest")

    def build_or_load(
        self,
        force_rebuild: bool = False,
    ) -> Chroma:
        with self._lock:
            current_digest = self._digest()
            can_reuse = (
                not force_rebuild
                and self.chroma_dir.is_dir()
                and self._manifest_digest() == current_digest
            )

            if can_reuse:
                self._vector_store = Chroma(
                    collection_name=COLLECTION_NAME,
                    embedding_function=self.embeddings,
                    persist_directory=str(self.chroma_dir),
                )
                return self._vector_store

            documents = self._load_documents()
            chunks = self._split_documents(documents)

            if self.chroma_dir.exists():
                shutil.rmtree(self.chroma_dir)

            self.chroma_dir.mkdir(parents=True, exist_ok=True)

            self._vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=str(self.chroma_dir),
                ids=[
                    str(chunk.metadata["chunk_id"])
                    for chunk in chunks
                ],
            )

            self.manifest_path.write_text(
                json.dumps(
                    {
                        "digest": current_digest,
                        "embedding_model": settings.embedding_model_name,
                        "chunk_size": settings.rag_chunk_size,
                        "chunk_overlap": settings.rag_chunk_overlap,
                        "document_count": len(documents),
                        "chunk_count": len(chunks),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            return self._vector_store

    @property
    def vector_store(self) -> Chroma:
        if self._vector_store is None:
            return self.build_or_load()
        return self._vector_store

    def search(
        self,
        question: str,
        disease_code: str | None = None,
        k: int | None = None,
    ) -> list[dict[str, Any]]:
        cleaned = question.strip()

        if not cleaned:
            raise ValueError("检索问题不能为空")

        top_k = settings.rag_top_k if k is None else k

        if not 1 <= top_k <= 12:
            raise ValueError("k必须在1到12之间")

        normalized_code = (
            disease_code.strip().lower()
            if disease_code
            else None
        )

        metadata_filter = (
            {"disease_code": normalized_code}
            if normalized_code
            else None
        )

        documents_with_scores = (
            self.vector_store.similarity_search_with_relevance_scores(
                query=cleaned,
                k=top_k,
                filter=metadata_filter,
            )
        )

        results: list[dict[str, Any]] = []

        for rank, (document, score) in enumerate(
            documents_with_scores,
            start=1,
        ):
            results.append(
                {
                    "rank": rank,
                    "relevance_score": round(float(score), 4),
                    "content": document.page_content,
                    "metadata": document.metadata,
                }
            )

        return results
