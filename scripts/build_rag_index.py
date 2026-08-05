from __future__ import annotations

import argparse
import json

from rice_agent.services.rag_store import RiceKnowledgeStore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    store = RiceKnowledgeStore()
    store.build_or_load(force_rebuild=args.force)

    manifest = store.manifest_path
    print(f"索引目录：{store.chroma_dir}")

    if manifest.is_file():
        print(json.dumps(
            json.loads(manifest.read_text(encoding="utf-8")),
            ensure_ascii=False,
            indent=2,
        ))
