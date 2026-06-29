import json
import logging
import os
from typing import Any

from model.embedding_llm import EMBEDDING_LLM
from qdrant.qdrantClient import QdrantManager

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "memory", "data")


class RAG_Manager:
    def __init__(
        self,
        collection_name: str = "rag_demo",
        vector_size: int = 1024,
        memory_mode: bool = False,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._embedding: EMBEDDING_LLM | None = None
        self._vector_db: QdrantManager | None = None
        self._memory_mode = memory_mode
        self._documents_path = os.path.join(DATA_DIR, f"{collection_name}_docs.json")
        logger.info(
            "RAG_Manager init | collection=%s size=%d memory_mode=%s",
            collection_name, vector_size, memory_mode,
        )

    # ── lazy init ─────────────────────────────────────────────────────

    @property
    def embedding(self) -> EMBEDDING_LLM:
        if self._embedding is None:
            try:
                self._embedding = EMBEDDING_LLM()
            except Exception as e:
                logger.error("embedding init failed: %s", e)
                raise
        return self._embedding

    @property
    def vector_db(self) -> QdrantManager:
        if self._vector_db is None:
            try:
                self._vector_db = QdrantManager(memory_mode=self._memory_mode)
                self._vector_db.create_collection(
                    collection_name=self.collection_name,
                    size=self.vector_size,
                )
            except Exception as e:
                logger.error("vector_db init failed: %s", e)
                raise
        return self._vector_db

    # ── document management ───────────────────────────────────────────

    def add_documents(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> int:
        if not texts:
            logger.warning("add_documents called with empty texts")
            return 0

        metadata = metadata or [{}] * len(texts)
        try:
            vectors = self.embedding.embbedding_vectors(texts)
            payloads = [
                {"content": text, **(metadata[i] if i < len(metadata) else {})}
                for i, text in enumerate(texts)
            ]
            self.vector_db.add_vectors(vectors=vectors, payloads=payloads)
            self._save_documents(texts, metadata)
            logger.info("added %d document(s) to RAG", len(texts))
            return len(texts)
        except Exception as e:
            logger.exception("add_documents failed: %s", e)
            return 0

    def query(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not question:
            return []
        try:
            vector = self.embedding.embbedding_vector(question)
            results = self.vector_db.query_vector(vector, top_k=top_k)
            logger.info("RAG query | q=%s hits=%d", question[:60], len(results))
            return results
        except Exception as e:
            logger.exception("RAG query failed: %s", e)
            return []

    def format_context(self, results: list[dict[str, Any]], max_chars: int = 2000) -> str:
        if not results:
            return ""
        parts: list[str] = []
        total = 0
        for r in results:
            content = r.get("payload", {}).get("content", "")
            score = r.get("score", 0)
            snippet = f"[score={score:.3f}] {content}"
            total += len(snippet)
            if total > max_chars:
                break
            parts.append(snippet)
        return "\n\n".join(parts)

    # ── persistence ───────────────────────────────────────────────────

    def _save_documents(self, texts: list[str], metadata: list[dict]) -> None:
        os.makedirs(os.path.dirname(self._documents_path), exist_ok=True)
        existing: list[dict] = []
        if os.path.exists(self._documents_path):
            try:
                with open(self._documents_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = []
        existing.extend({"content": t, "metadata": m} for t, m in zip(texts, metadata))
        try:
            with open(self._documents_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("save documents failed: %s", e)

    def clear(self) -> None:
        try:
            self.vector_db.delete_collection()
            self.vector_db.create_collection(
                collection_name=self.collection_name,
                size=self.vector_size,
            )
        except Exception as e:
            logger.warning("clear RAG vector_db failed: %s", e)
        if os.path.exists(self._documents_path):
            try:
                os.remove(self._documents_path)
            except Exception as e:
                logger.warning("clear RAG docs file failed: %s", e)
        logger.info("RAG cleared")
