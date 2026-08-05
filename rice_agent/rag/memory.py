from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class ConversationTurn:
    user: str
    assistant: str
    turn_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SlidingWindowConversationStore:
    """内存会话存储；提示词只读取最近 N 轮和限定字符数。"""

    def __init__(self, max_turns: int = 6, max_chars: int = 12000) -> None:
        if max_turns < 1 or max_chars < 256:
            raise ValueError("滑动窗口参数无效")
        self.max_turns = max_turns
        self.max_chars = max_chars
        self._sessions: dict[str, deque[ConversationTurn]] = {}
        self._lock = Lock()

    def create_session(self) -> str:
        session_id = uuid4().hex
        with self._lock:
            self._sessions[session_id] = deque(maxlen=self.max_turns * 4)
        return session_id

    def ensure_session(self, session_id: str | None) -> str:
        cleaned = (session_id or "").strip()
        with self._lock:
            if cleaned and cleaned in self._sessions:
                return cleaned
            new_id = cleaned or uuid4().hex
            self._sessions[new_id] = deque(maxlen=self.max_turns * 4)
            return new_id

    def append(self, session_id: str, turn: ConversationTurn) -> None:
        with self._lock:
            bucket = self._sessions.setdefault(
                session_id,
                deque(maxlen=self.max_turns * 4),
            )
            bucket.append(turn)

    def history(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            turns = list(self._sessions.get(session_id, ()))

        selected: list[ConversationTurn] = []
        used_chars = 0
        for turn in reversed(turns[-self.max_turns :]):
            turn_chars = len(turn.user) + len(turn.assistant)
            if selected and used_chars + turn_chars > self.max_chars:
                break
            selected.append(turn)
            used_chars += turn_chars

        selected.reverse()
        history: list[dict[str, str]] = []
        for turn in selected:
            history.extend(
                [
                    {"role": "user", "content": turn.user},
                    {"role": "assistant", "content": turn.assistant},
                ]
            )
        return history

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
