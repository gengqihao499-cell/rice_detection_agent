from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rice_agent.config import PROJECT_ROOT, settings
from rice_agent.domain import CODE_METADATA


COLLECTION_NAME = "rice_disease_knowledge"
INDEX_SCHEMA_VERSION = "parent-child-v2-offset-mapping"
_TOKEN_FALLBACK = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+|[^\s]")


def _default_query_prompt(model_name: str) -> str:
    lowered = model_name.casefold()
    if "qwen3-embedding" in lowered:
        return (
            "Instruct: Given a query about rice diseases, pests, symptoms, "
            "diagnosis, and crop management, retrieve relevant passages that "
            "answer the query\nQuery: "
        )
    if "bge-" in lowered and "zh-v1.5" in lowered:
        return "为这个句子生成表示以用于检索相关文章："
    return ""


class RiceKnowledgeStore:
    """父子分块知识库：子块向量召回，命中后回填父块。"""

    def __init__(self) -> None:
        self.knowledge_dir = settings.knowledge_dir
        self.corpus_dir = settings.knowledge_corpus_dir
        self.chroma_dir = settings.chroma_dir
        self.manifest_path = self.chroma_dir / "manifest.json"
        self.parent_store_path = self.chroma_dir / "parents.jsonl"
        self._embeddings: HuggingFaceEmbeddings | None = None
        self._vector_store: Chroma | None = None
        self._tokenizer: Any | None = None
        self._tokenizer_attempted = False
        self._parents: dict[str, Document] = {}
        self._chunks: list[Document] | None = None
        self._lock = Lock()

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            query_prompt = (
                settings.embedding_query_prompt
                or _default_query_prompt(settings.embedding_model_name)
            )
            query_encode_kwargs: dict[str, Any] = {
                "normalize_embeddings": True,
            }
            if query_prompt:
                query_encode_kwargs["prompt"] = query_prompt
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model_name,
                model_kwargs={
                    "device": settings.embedding_device,
                    "local_files_only": settings.embedding_local_files_only,
                },
                encode_kwargs={"normalize_embeddings": True},
                query_encode_kwargs=query_encode_kwargs,
            )
        return self._embeddings

    def _knowledge_files(self) -> list[Path]:
        files = sorted(self.knowledge_dir.glob("*.md"))
        if self.corpus_dir.is_dir():
            files.extend(sorted(self.corpus_dir.rglob("*.txt")))
            files.extend(sorted(self.corpus_dir.rglob("*.md")))
        files = [
            path
            for path in files
            if path.name not in {"SOURCE_CATALOG.md"}
        ]
        if not files:
            raise FileNotFoundError(
                f"知识目录中没有可索引文本：{self.knowledge_dir} / {self.corpus_dir}"
            )
        return files

    def _source_manifest(self) -> dict[str, dict[str, Any]]:
        manifest_path = self.corpus_dir / "manifest.jsonl"
        result: dict[str, dict[str, Any]] = {}
        if not manifest_path.is_file():
            return result
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            filename = str(item.get("file") or "")
            if filename:
                result[filename] = item
        return result

    def _digest(self) -> str:
        digest = hashlib.sha256()
        for value in (
            INDEX_SCHEMA_VERSION,
            settings.embedding_model_name,
            settings.embedding_query_prompt,
            settings.rag_parent_chunk_tokens,
            settings.rag_parent_overlap_tokens,
            settings.rag_child_chunk_tokens,
            settings.rag_child_overlap_tokens,
            settings.rag_tokenizer_model_name,
        ):
            digest.update(str(value).encode("utf-8"))
        for path in self._knowledge_files():
            try:
                relative = path.resolve().relative_to(PROJECT_ROOT)
            except ValueError:
                relative = path.resolve()
            digest.update(str(relative).encode("utf-8"))
            digest.update(path.read_bytes())
        source_manifest = self.corpus_dir / "manifest.jsonl"
        if source_manifest.is_file():
            digest.update(source_manifest.read_bytes())
        return digest.hexdigest()

    def _load_documents(self) -> list[Document]:
        documents: list[Document] = []
        source_manifest = self._source_manifest()
        for path in self._knowledge_files():
            content = path.read_text(encoding="utf-8", errors="replace").strip()
            if not content:
                continue
            code = path.stem
            domain = CODE_METADATA.get(code, {})
            external = source_manifest.get(path.name, {})
            source_id = str(external.get("source_id") or code)
            metadata = {
                "disease_code": code if code in CODE_METADATA else "",
                "display_name_zh": domain.get(
                    "display_name_zh",
                    external.get("title") or code,
                ),
                "kind": domain.get(
                    "kind",
                    "open_access_research" if external else "unknown",
                ),
                "source": path.name,
                "source_id": source_id,
                "source_path": str(path.resolve()),
                "source_url": str(external.get("source_url") or ""),
                "license": str(external.get("license") or ""),
                "license_url": str(external.get("license_url") or ""),
                "doi": str(external.get("doi") or ""),
                "provider": str(external.get("provider") or "RiceCare curated"),
                "title": str(external.get("title") or domain.get("display_name_zh") or code),
            }
            documents.append(Document(page_content=content, metadata=metadata))
        if not documents:
            raise RuntimeError("知识文档均为空")
        return documents

    def _get_tokenizer(self) -> Any | None:
        if self._tokenizer_attempted:
            return self._tokenizer
        self._tokenizer_attempted = True
        model_name = (
            settings.rag_tokenizer_model_name
            or settings.embedding_model_name
        )
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=settings.embedding_local_files_only,
            )
            # 此 tokenizer 仅用于整篇文档的长度计算，不执行模型前向；放宽告警阈值，
            # 真正送入嵌入模型的文本仍由 child splitter 限制为 150 token。
            self._tokenizer.model_max_length = max(
                int(getattr(self._tokenizer, "model_max_length", 0) or 0),
                1_000_000_000,
            )
        except Exception:
            self._tokenizer = None
        return self._tokenizer

    def _token_length_function(self) -> tuple[Callable[[str], int], str]:
        tokenizer = self._get_tokenizer()
        if tokenizer is not None:
            return (
                lambda text: len(
                    tokenizer.encode(text, add_special_tokens=False)
                ),
                f"hf:{getattr(tokenizer, 'name_or_path', 'unknown')}",
            )
        return (
            lambda text: len(_TOKEN_FALLBACK.findall(text)),
            "regex_approximation",
        )

    @staticmethod
    def _separators() -> list[str]:
        return [
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            ". ",
            "! ",
            "? ",
            "; ",
            "，",
            ", ",
            " ",
            "",
        ]

    def _split_documents_with_tokenizer(
        self,
        documents: list[Document],
        tokenizer: Any,
    ) -> list[Document]:
        """每篇只 tokenize 一次，再按 token ID 窗口建立严格父子块。"""
        parent_size = settings.rag_parent_chunk_tokens
        parent_step = max(1, parent_size - settings.rag_parent_overlap_tokens)
        child_size = settings.rag_child_chunk_tokens
        child_step = max(1, child_size - settings.rag_child_overlap_tokens)
        token_method = f"hf:{getattr(tokenizer, 'name_or_path', 'unknown')}"
        children: list[Document] = []
        self._parents = {}

        for document in documents:
            encoded = tokenizer(
                document.page_content,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False,
            )
            token_ids = list(encoded["input_ids"])
            offsets = list(encoded.get("offset_mapping") or [])

            def original_window(start: int, end: int, ids: list[int]) -> str:
                if offsets and start < len(offsets) and end <= len(offsets):
                    char_start = int(offsets[start][0])
                    char_end = int(offsets[end - 1][1])
                    if char_end > char_start:
                        return document.page_content[char_start:char_end].strip()
                return tokenizer.decode(
                    ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()

            source_id = str(document.metadata.get("source_id") or "source")
            parent_index = 0
            for parent_start in range(0, len(token_ids), parent_step):
                parent_ids = token_ids[parent_start : parent_start + parent_size]
                if not parent_ids:
                    break
                parent_content = original_window(
                    parent_start,
                    parent_start + len(parent_ids),
                    parent_ids,
                )
                if not parent_content:
                    continue
                identity = f"{source_id}|{parent_start}|{parent_ids[:32]}"
                parent_id = hashlib.sha1(
                    identity.encode("utf-8")
                ).hexdigest()[:24]
                parent_metadata = {
                    **document.metadata,
                    "parent_id": parent_id,
                    "chunk_id": parent_id,
                    "parent_index": parent_index,
                    "parent_start_token": parent_start,
                    "chunk_level": "parent",
                    "tokenizer_method": token_method,
                    "token_count": len(parent_ids),
                }
                parent_document = Document(
                    page_content=parent_content,
                    metadata=parent_metadata,
                )
                self._parents[parent_id] = parent_document

                for child_index, child_start in enumerate(
                    range(0, len(parent_ids), child_step)
                ):
                    child_ids = parent_ids[child_start : child_start + child_size]
                    if not child_ids:
                        break
                    global_child_start = parent_start + child_start
                    child_content = original_window(
                        global_child_start,
                        global_child_start + len(child_ids),
                        child_ids,
                    )
                    if not child_content:
                        continue
                    child_id = f"{parent_id}_c{child_index:03d}"
                    child_metadata = {
                        **parent_metadata,
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "child_index": child_index,
                        "child_start_token": global_child_start,
                        "chunk_level": "child",
                        "token_count": len(child_ids),
                    }
                    children.append(
                        Document(
                            page_content=child_content,
                            metadata=child_metadata,
                        )
                    )
                parent_index += 1
                if parent_start + parent_size >= len(token_ids):
                    break
        return children

    def _split_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        tokenizer = self._get_tokenizer()
        if tokenizer is not None:
            return self._split_documents_with_tokenizer(documents, tokenizer)
        length_function, token_method = self._token_length_function()
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_parent_chunk_tokens,
            chunk_overlap=settings.rag_parent_overlap_tokens,
            separators=self._separators(),
            keep_separator=True,
            add_start_index=True,
            length_function=length_function,
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_child_chunk_tokens,
            chunk_overlap=settings.rag_child_overlap_tokens,
            separators=self._separators(),
            keep_separator=True,
            add_start_index=True,
            length_function=length_function,
        )
        children: list[Document] = []
        self._parents = {}

        for document in documents:
            source_id = str(document.metadata.get("source_id") or "source")
            parents = parent_splitter.split_documents([document])
            for parent_index, parent in enumerate(parents):
                parent_start = int(parent.metadata.get("start_index", 0) or 0)
                identity = (
                    f"{source_id}|{parent_index}|{parent_start}|"
                    f"{parent.page_content[:160]}"
                )
                parent_id = hashlib.sha1(
                    identity.encode("utf-8")
                ).hexdigest()[:24]
                parent_metadata = {
                    **parent.metadata,
                    "parent_id": parent_id,
                    "chunk_id": parent_id,
                    "parent_index": parent_index,
                    "parent_start_index": parent_start,
                    "chunk_level": "parent",
                    "tokenizer_method": token_method,
                    "token_count": length_function(parent.page_content),
                }
                parent_document = Document(
                    page_content=parent.page_content,
                    metadata=parent_metadata,
                )
                self._parents[parent_id] = parent_document

                child_documents = child_splitter.split_documents(
                    [parent_document]
                )
                for child_index, child in enumerate(child_documents):
                    child_id = f"{parent_id}_c{child_index:03d}"
                    child_start = int(
                        child.metadata.get("start_index", 0) or 0
                    )
                    child.metadata = {
                        **parent_metadata,
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "child_index": child_index,
                        "child_start_index": child_start,
                        "chunk_level": "child",
                        "token_count": length_function(child.page_content),
                    }
                    children.append(child)
        return children

    def load_chunks(self) -> list[Document]:
        """返回约150-token子块，供向量索引、BM25和评测复用。"""
        if self._chunks is None:
            self._chunks = self._split_documents(self._load_documents())
        return list(self._chunks)

    def load_parents(self) -> dict[str, Document]:
        if self._parents:
            return dict(self._parents)
        if not self.parent_store_path.is_file():
            self.load_chunks()
            return dict(self._parents)
        for line in self.parent_store_path.read_text(
            encoding="utf-8"
        ).splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            parent_id = str(item.get("parent_id") or "")
            if parent_id:
                self._parents[parent_id] = Document(
                    page_content=str(item.get("content") or ""),
                    metadata=dict(item.get("metadata") or {}),
                )
        return dict(self._parents)

    def _write_parent_store(self) -> None:
        with self.parent_store_path.open("w", encoding="utf-8") as stream:
            for parent_id, document in self._parents.items():
                stream.write(
                    json.dumps(
                        {
                            "parent_id": parent_id,
                            "content": document.page_content,
                            "metadata": document.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

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
                and self.parent_store_path.is_file()
                and self._manifest_digest() == current_digest
            )
            if can_reuse:
                self.load_parents()
                self._vector_store = Chroma(
                    collection_name=COLLECTION_NAME,
                    embedding_function=self.embeddings,
                    persist_directory=str(self.chroma_dir),
                )
                return self._vector_store

            self._chunks = None
            self._parents = {}
            documents = self._load_documents()
            children = self._split_documents(documents)
            self._chunks = children
            if self.chroma_dir.exists():
                shutil.rmtree(self.chroma_dir)
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._write_parent_store()
            self._vector_store = Chroma.from_documents(
                documents=children,
                embedding=self.embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=str(self.chroma_dir),
                ids=[
                    str(child.metadata["chunk_id"])
                    for child in children
                ],
            )
            parent_token_counts = [
                int(document.metadata.get("token_count", 0) or 0)
                for document in self._parents.values()
            ]
            child_token_counts = [
                int(document.metadata.get("token_count", 0) or 0)
                for document in children
            ]
            self.manifest_path.write_text(
                json.dumps(
                    {
                        "digest": current_digest,
                        "index_schema_version": INDEX_SCHEMA_VERSION,
                        "embedding_model": settings.embedding_model_name,
                        "parent_chunk_tokens": settings.rag_parent_chunk_tokens,
                        "parent_overlap_tokens": settings.rag_parent_overlap_tokens,
                        "child_chunk_tokens": settings.rag_child_chunk_tokens,
                        "child_overlap_tokens": settings.rag_child_overlap_tokens,
                        "document_count": len(documents),
                        "parent_count": len(self._parents),
                        "child_count": len(children),
                        "max_parent_tokens": max(parent_token_counts, default=0),
                        "max_child_tokens": max(child_token_counts, default=0),
                        "tokenizer_method": next(
                            iter(children),
                            Document(page_content="", metadata={}),
                        ).metadata.get("tokenizer_method", ""),
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

    def expand_to_parents(
        self,
        results: list[dict[str, Any]],
        k: int | None = None,
    ) -> list[dict[str, Any]]:
        parents = self.load_parents()
        merged: dict[str, dict[str, Any]] = {}
        for result in results:
            metadata = dict(result.get("metadata") or {})
            parent_id = str(metadata.get("parent_id") or "")
            parent = parents.get(parent_id)
            if not parent:
                key = str(metadata.get("chunk_id") or result.get("content") or "")
                if key:
                    merged[key] = dict(result)
                continue
            score = float(result.get("relevance_score", 0.0) or 0.0)
            if parent_id not in merged:
                merged[parent_id] = {
                    **result,
                    "content": parent.page_content,
                    "metadata": {
                        **parent.metadata,
                        "chunk_id": parent_id,
                        "parent_id": parent_id,
                        "matching_child_ids": [],
                    },
                    "matching_children": [],
                }
            item = merged[parent_id]
            item["relevance_score"] = max(
                score,
                float(item.get("relevance_score", 0.0) or 0.0),
            )
            child_id = str(metadata.get("chunk_id") or "")
            if child_id and child_id not in item["metadata"]["matching_child_ids"]:
                item["metadata"]["matching_child_ids"].append(child_id)
            if len(item["matching_children"]) < 3:
                item["matching_children"].append(
                    {
                        "chunk_id": child_id,
                        "content": str(result.get("content") or ""),
                        "score": score,
                    }
                )
        ordered = sorted(
            merged.values(),
            key=lambda item: float(item.get("relevance_score", 0.0) or 0.0),
            reverse=True,
        )
        if k is not None:
            ordered = ordered[:k]
        for rank, item in enumerate(ordered, 1):
            item["rank"] = rank
            item["relevance_score"] = round(
                float(item.get("relevance_score", 0.0) or 0.0),
                4,
            )
        return ordered

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
        if not 1 <= top_k <= 24:
            raise ValueError("k必须在1到24之间")
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
        child_k = min(96, max(top_k * 4, top_k))
        documents_with_scores = (
            self.vector_store.similarity_search_with_score(
                query=cleaned,
                k=child_k,
                filter=metadata_filter,
            )
        )
        child_results: list[dict[str, Any]] = []
        for rank, (document, distance) in enumerate(
            documents_with_scores,
            start=1,
        ):
            # Chroma 默认返回平方 L2 距离；向量归一化时 cosine=1-distance/2。
            score = max(0.0, min(1.0, 1.0 - float(distance) / 2.0))
            child_results.append(
                {
                    "rank": rank,
                    "relevance_score": round(float(score), 4),
                    "content": document.page_content,
                    "metadata": document.metadata,
                }
            )
        return self.expand_to_parents(child_results, top_k)
