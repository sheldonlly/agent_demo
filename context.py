import json
import logging
import os
from datetime import datetime
from typing import Any

from memory.memoryManager import MemoryManager
from RAG.rag import RAG_Manager

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory", "data")
CONTEXT_PATH = os.path.join(DATA_DIR, "context_history.json")

MAX_HISTORY_TURNS = 20
MAX_HISTORY_CHARS = 2000


class ContextManager:
    def __init__(
        self,
        memory: MemoryManager | None = None,
        rag: RAG_Manager | None = None,
    ) -> None:
        self.memory = memory or MemoryManager()
        self.rag = rag
        self._history: list[dict[str, Any]] = []
        self._pending_actions: list[dict[str, Any]] = []
        self._max_turns = MAX_HISTORY_TURNS
        self._load_history()
        logger.info("ContextManager ready | history=%d turns", len(self._history))

    # ── conversation history ───────────────────────────────────────────

    def add_turn(self, role: str, content: str) -> None:
        if not content:
            return
        turn = {
            "role": role,
            "content": content[:2000],
            "timestamp": datetime.now().isoformat(),
        }
        self._history.append(turn)
        if len(self._history) > self._max_turns:
            self._summarize_and_prune()
        self._save_history()

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        if limit:
            return self._history[-limit:]
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()
        self._save_history()
        logger.info("conversation history cleared")

    # ── context assembly ───────────────────────────────────────────────

    def build_prompt(
        self,
        query: str,
        system_instruction: str = "",
        use_rag: bool = True,
    ) -> str:
        parts: list[str] = []

        if system_instruction:
            parts.append(f"[System]\n{system_instruction}")

        recent = self.memory.get_recent_context(limit=5)
        if recent:
            parts.append(f"[Recent Memory]\n{recent}")

        if use_rag and self.rag:
            try:
                rag_results = self.rag.query(query, top_k=2)
                if rag_results:
                    context = self.rag.format_context(rag_results, max_chars=1500)
                    if context:
                        parts.append(f"[Knowledge]\n{context}")
            except Exception as e:
                logger.warning("RAG query failed in build_prompt: %s", e)

        history = self.get_history(limit=6)
        if history:
            hist_lines: list[str] = []
            total = 0
            for h in history:
                line = f"{h['role'].capitalize()}: {h['content'][:300]}"
                total += len(line)
                if total > MAX_HISTORY_CHARS:
                    break
                hist_lines.append(line)
            parts.append(f"[Conversation]\n" + "\n".join(hist_lines))

        parts.append(f"[Query]\n{query}")

        return "\n\n".join(parts)

    # ── human-in-the-loop ──────────────────────────────────────────────

    def request_approval(self, action: dict[str, Any]) -> bool:
        self._pending_actions.append(action)
        logger.warning(
            "Approval required | action=%s args=%s",
            action.get("name"),
            str(action.get("args", {}))[:200],
        )
        return False

    def approve_last_action(self) -> bool:
        if not self._pending_actions:
            return False
        self._pending_actions.pop()
        return True

    def reject_last_action(self) -> bool:
        if not self._pending_actions:
            return False
        rejected = self._pending_actions.pop()
        logger.info("Action rejected: %s", rejected.get("name"))
        return True

    def pending_count(self) -> int:
        return len(self._pending_actions)

    # ── recording ──────────────────────────────────────────────────────

    def record_interaction(self, query: str, response: str) -> None:
        self.add_turn("user", query)
        self.add_turn("assistant", response)
        self.memory.record_interaction(query, response)
        logger.debug("recorded interaction | q=%s", query[:60])

    # ── persistence ────────────────────────────────────────────────────

    def _summarize_and_prune(self) -> None:
        early = self._history[: len(self._history) // 2]
        summary = f"[Session summary: {len(early)} earlier turns]"
        self._history = (
            [{"role": "system", "content": summary, "timestamp": datetime.now().isoformat()}]
            + self._history[len(self._history) // 2 :]
        )
        logger.info("history pruned from %d to %d turns", len(self._history) + len(early), len(self._history))

    def _load_history(self) -> None:
        if not os.path.exists(CONTEXT_PATH):
            return
        try:
            with open(CONTEXT_PATH, "r", encoding="utf-8") as f:
                self._history = json.load(f)
        except Exception as e:
            logger.warning("load history failed: %s", e)
            self._history = []

    def _save_history(self) -> None:
        os.makedirs(os.path.dirname(CONTEXT_PATH), exist_ok=True)
        try:
            with open(CONTEXT_PATH, "w", encoding="utf-8") as f:
                json.dump(self._history[-self._max_turns :], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("save history failed: %s", e)
