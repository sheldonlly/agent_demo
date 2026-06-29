import logging
from memory.memory import WorkMemory, SemanticMemory, EpisodicMemory, PerceptualMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self) -> None:
        self.work = WorkMemory()
        self.semantic = SemanticMemory()
        self.episodic = EpisodicMemory()
        self.perceptual = PerceptualMemory()
        logger.info("MemoryManager ready")

    # ── recording ──────────────────────────────────────────────────────

    def record_interaction(self, query: str, response: str) -> None:
        if not query or not response:
            return
        self.work.add(f"Q: {query}\nA: {response[:500]}", {"type": "interaction"})
        self.episodic.add(f"User: {query}\nAssistant: {response[:1000]}", {"type": "conversation"})
        logger.debug("recorded interaction | query_len=%d resp_len=%d", len(query), len(response))

    def record_fact(self, fact: str, source: str = "") -> None:
        self.semantic.add(fact, {"type": "fact", "source": source})

    def record_observation(self, observation: str) -> None:
        self.perceptual.add(observation, {"type": "observation"})

    # ── retrieval ─────────────────────────────────────────────────────

    def get_recent_context(self, limit: int = 5) -> str:
        return self.work.get_context(limit)

    def search_all(self, query: str, limit: int = 3) -> dict[str, list[str]]:
        return {
            "work": [it.content[:200] for it in self.work.search(query, limit)],
            "semantic": [it.content for it in self.semantic.search(query, limit)],
            "episodic": [it.content[:200] for it in self.episodic.search(query, limit)],
            "perceptual": [it.content[:200] for it in self.perceptual.search(query, limit)],
        }

    def stats(self) -> dict[str, int]:
        return {
            "work": self.work.count(),
            "semantic": self.semantic.count(),
            "episodic": self.episodic.count(),
            "perceptual": self.perceptual.count(),
        }

    def clear_all(self) -> None:
        self.work.clear()
        self.semantic.clear()
        self.episodic.clear()
        self.perceptual.clear()
        logger.info("all memories cleared")
