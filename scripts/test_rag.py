from __future__ import annotations

import argparse
import json

from rice_agent.services.rag_store import RiceKnowledgeStore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question",
        nargs="?",
        default="有哪些典型症状和基础管理建议？",
    )
    parser.add_argument(
        "--code",
        default="leaf_blast",
    )
    args = parser.parse_args()

    results = RiceKnowledgeStore().search(
        question=args.question,
        disease_code=args.code,
        k=4,
    )
    print(json.dumps(
        results,
        ensure_ascii=False,
        indent=2,
    ))
