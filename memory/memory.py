import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@dataclass
class MemoryItem:
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)


class BaseMemory:
    def __init__(self, name: str) -> None:
        self.name = name
        self._path = os.path.join(DATA_DIR, f"{name}.json")
        self._items: list[MemoryItem] = []
        self._load()

    # ── public API ────────────────────────────────────────────────────

    def add(self, content: str, metadata: dict | None = None) -> None:
        if not content or not content.strip():
            logger.warning("[%s] refusing empty content", self.name)
            return
        item = MemoryItem(content=content, metadata=metadata or {})
        self._items.append(item)
        self._save()
        logger.debug("[%s] added item | len=%d | total=%d", self.name, len(content), len(self._items))

    def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        q = query.lower()
        results = [it for it in self._items if q in it.content.lower()]
        return results[-limit:]

    def clear(self) -> None:
        self._items.clear()
        self._save()
        logger.info("[%s] cleared", self.name)

    def get_all(self) -> list[MemoryItem]:
        return list(self._items)

    def count(self) -> int:
        return len(self._items)

    # ── persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._items = [MemoryItem(**it) for it in data]
            logger.info("[%s] loaded %d item(s) from %s", self.name, len(self._items), self._path)
        except Exception as e:
            logger.warning("[%s] load failed: %s", self.name, e)

    def _save(self) -> None:
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump([asdict(it) for it in self._items], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("[%s] save failed: %s", self.name, e)


class WorkMemory(BaseMemory):
    def __init__(self) -> None:
        super().__init__("work_memory")

    def get_context(self, limit: int = 10) -> str:
        recent = self._items[-limit:]
        if not recent:
            return ""
        return "\n".join(f"- {it.content[:200]}" for it in recent)


class SemanticMemory(BaseMemory):
    def __init__(self) -> None:
        super().__init__("semantic_memory")


class EpisodicMemory(BaseMemory):
    def __init__(self) -> None:
        super().__init__("episodic_memory")


class PerceptualMemory(BaseMemory):
    def __init__(self) -> None:
        super().__init__("perceptual_memory")
