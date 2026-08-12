from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rice_agent.config import settings
from rice_agent.services.rag_store import RiceKnowledgeStore
from scripts.download_open_access_corpus import DEFAULT_QUERY, download_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按配置下载开放许可语料、构建父子 Chroma 索引，并可直接启动服务。"
    )
    parser.add_argument("--target-mb", type=float, default=settings.knowledge_target_mb)
    parser.add_argument("--output-dir", type=Path, default=settings.knowledge_corpus_dir)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--max-articles", type=int, default=5000)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--email", default=os.getenv("EUROPE_PMC_EMAIL", ""))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    report: dict[str, object] = {
        "configuration": {
            "embedding_model": settings.embedding_model_name,
            "parent_chunk_tokens": settings.rag_parent_chunk_tokens,
            "child_chunk_tokens": settings.rag_child_chunk_tokens,
            "target_mb": args.target_mb,
        }
    }
    if not args.skip_download:
        report["download"] = download_corpus(
            output_dir=args.output_dir.resolve(),
            target_mb=args.target_mb,
            query=args.query,
            email=args.email,
            max_articles=args.max_articles,
            delay=max(0.0, args.delay),
            workers=args.workers,
        )

    if not args.skip_index:
        store = RiceKnowledgeStore()
        store.build_or_load(force_rebuild=args.force_index)
        report["index"] = json.loads(
            store.manifest_path.read_text(encoding="utf-8")
        )
        report["index_directory"] = str(store.chroma_dir)

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.serve:
        import uvicorn

        uvicorn.run("rice_agent.web.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
