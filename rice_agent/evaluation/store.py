from __future__ import annotations

import asyncio
import json
from pathlib import Path
from threading import Lock
from typing import Any


class JsonlQualityStore:
    def __init__(self, evaluation_path: Path, feedback_path: Path) -> None:
        self.evaluation_path = evaluation_path
        self.feedback_path = feedback_path
        self._write_lock = Lock()

    async def append_evaluation(self, payload: dict[str, Any]) -> None:
        await asyncio.to_thread(self._append, self.evaluation_path, payload)

    async def append_feedback(self, payload: dict[str, Any]) -> None:
        await asyncio.to_thread(self._append, self.feedback_path, payload)

    def _append(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self._write_lock:
            with path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
