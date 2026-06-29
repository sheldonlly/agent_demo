import json
import logging
import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from memory.memoryManager import MemoryManager
from RAG.rag import RAG_Manager

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory", "data")
CONTEXT_PATH = os.path.join(DATA_DIR, "context_history.json")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")

MAX_HISTORY_TURNS = 20
MAX_HISTORY_CHARS = 2000


class SessionPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class ContextManager:
    def __init__(
        self,
        memory: MemoryManager | None = None,
        rag: RAG_Manager | None = None,
        session_id: str | None = None,
    ) -> None:
        self.memory = memory or MemoryManager()
        self.rag = rag
        self._history: list[dict[str, Any]] = []
        self._pending_actions: list[dict[str, Any]] = []
        self._max_turns = MAX_HISTORY_TURNS
        self._session_id = session_id or self._generate_session_id()
        self._session_meta: dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "turn_count": 0,
            "priority": SessionPriority.NORMAL.name,
            "tags": [],
        }
        self._load_history()
        self._load_session_meta()
        logger.info(
            "ContextManager ready | session=%s history=%d turns",
            self._session_id, len(self._history),
        )

    # ── session management ─────────────────────────────────────────────

    @staticmethod
    def _generate_session_id() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]

    @property
    def session_id(self) -> str:
        return self._session_id

    def set_session_priority(self, priority: SessionPriority) -> None:
        self._session_meta["priority"] = priority.name
        self._save_session_meta()
        logger.info("session %s priority set to %s", self._session_id, priority.name)

    def add_session_tag(self, tag: str) -> None:
        if tag and tag not in self._session_meta["tags"]:
            self._session_meta["tags"].append(tag)
            self._save_session_meta()
            logger.debug("session %s tag added: %s", self._session_id, tag)

    def get_session_info(self) -> dict[str, Any]:
        return {
            "session_id": self._session_id,
            **self._session_meta,
            "history_turns": len(self._history),
            "pending_actions": len(self._pending_actions),
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        sessions = []
        for fname in os.listdir(SESSIONS_DIR):
            if fname.endswith("_meta.json"):
                try:
                    with open(os.path.join(SESSIONS_DIR, fname), "r", encoding="utf-8") as f:
                        data = json.load(f)
                    sid = fname.replace("_meta.json", "")
                    sessions.append({"session_id": sid, **data})
                except Exception as e:
                    logger.warning("failed to load session %s: %s", fname, e)
        return sorted(sessions, key=lambda x: x.get("created_at", ""), reverse=True)

    def switch_session(self, session_id: str) -> bool:
        meta_path = os.path.join(SESSIONS_DIR, f"{session_id}_meta.json")
        if not os.path.exists(meta_path):
            logger.warning("session %s not found", session_id)
            return False
        self._save_session_meta()
        self._session_id = session_id
        self._load_session_meta()
        self._history = []
        self._load_history()
        logger.info("switched to session %s | turns=%d", session_id, len(self._history))
        return True

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
        self._session_meta["turn_count"] += 1
        self._session_meta["updated_at"] = datetime.now().isoformat()
        if len(self._history) > self._max_turns:
            self._summarize_and_prune()
        self._save_history()
        self._save_session_meta()

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        if limit:
            return self._history[-limit:]
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()
        self._save_history()
        logger.info("conversation history cleared for session %s", self._session_id)

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
            "Approval required | action=%s args=%s session=%s",
            action.get("name"),
            str(action.get("args", {}))[:200],
            self._session_id,
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

    def get_pending_actions(self) -> list[dict[str, Any]]:
        return list(self._pending_actions)

    # ── recording ──────────────────────────────────────────────────────

    def record_interaction(self, query: str, response: str) -> None:
        self.add_turn("user", query)
        self.add_turn("assistant", response)
        self.memory.record_interaction(query, response)
        logger.debug("recorded interaction | session=%s q=%s", self._session_id, query[:60])

    # ── export ─────────────────────────────────────────────────────────

    def export_history(self, format: str = "json") -> str:
        if format == "markdown":
            lines = [f"# Session: {self._session_id}\n"]
            for h in self._history:
                lines.append(f"**{h['role'].capitalize()}** ({h['timestamp']}):")
                lines.append(f"{h['content']}\n")
            return "\n".join(lines)
        return json.dumps(self._history, ensure_ascii=False, indent=2)

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
        history_path = os.path.join(SESSIONS_DIR, f"{self._session_id}_history.json")
        if not os.path.exists(history_path):
            return
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                self._history = json.load(f)
        except Exception as e:
            logger.warning("load history for session %s failed: %s", self._session_id, e)
            self._history = []

    def _save_history(self) -> None:
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        history_path = os.path.join(SESSIONS_DIR, f"{self._session_id}_history.json")
        try:
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self._history[-self._max_turns :], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("save history for session %s failed: %s", self._session_id, e)

    def _save_session_meta(self) -> None:
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        meta_path = os.path.join(SESSIONS_DIR, f"{self._session_id}_meta.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self._session_meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("save session meta failed: %s", e)

    def _load_session_meta(self) -> None:
        meta_path = os.path.join(SESSIONS_DIR, f"{self._session_id}_meta.json")
        if not os.path.exists(meta_path):
            return
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                self._session_meta = json.load(f)
        except Exception as e:
            logger.warning("load session meta failed: %s", e)
