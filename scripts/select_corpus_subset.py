from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any


def select_subset(
    source_dir: Path,
    output_dir: Path,
    target_mb: float,
    seed: int = 42,
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if source_dir == output_dir:
        raise ValueError("source_dir 与 output_dir 不能相同")
    manifest_path = source_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到来源 manifest：{manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in output_dir.iterdir() if path.name != ".gitkeep"]
    if existing:
        raise FileExistsError(f"输出目录必须为空：{output_dir}")

    rows: list[dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        filename = str(item.get("file") or "")
        if filename and (source_dir / filename).is_file():
            rows.append(item)
    random.Random(seed).shuffle(rows)

    target_bytes = int(target_mb * 1024 * 1024)
    selected: list[dict[str, Any]] = []
    total_bytes = 0
    for item in rows:
        source_file = source_dir / str(item["file"])
        shutil.copy2(source_file, output_dir / source_file.name)
        selected.append(item)
        total_bytes += source_file.stat().st_size
        if total_bytes >= target_bytes:
            break

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for item in selected:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {
        "source_articles": len(rows),
        "selected_articles": len(selected),
        "target_bytes": target_bytes,
        "selected_bytes": total_bytes,
        "selected_mib": round(total_bytes / 1024 / 1024, 3),
        "seed": seed,
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="从已下载语料中确定性抽取指定大小子集。")
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--target-mb", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = select_subset(
        args.source_dir,
        args.output_dir,
        args.target_mb,
        args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
