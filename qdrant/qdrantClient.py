import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

import log.logconfig  # noqa: F401

logger = logging.getLogger(__name__)


class QdrantManager:
    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 6333,
        timeout: int = 30,
        memory_mode: bool = False,
    ) -> None:
        self.collection_name: str | None = None
        if memory_mode:
            self.client = QdrantClient(":memory:")
            logger.info("QdrantManager running in-memory")
        else:
            url = f"http://{ip}:{port}"
            self.client = QdrantClient(url=url, timeout=timeout)
            logger.info("QdrantManager connected to %s", url)

    # ── connection ────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    # ── collection ────────────────────────────────────────────────────

    def create_collection(
        self,
        collection_name: str = "sheldonDemo",
        size: int = 1024,
        distance: Distance = Distance.COSINE,
    ) -> None:
        self.collection_name = collection_name
        if self.client.collection_exists(collection_name):
            logger.debug("collection %s already exists", collection_name)
            return
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=distance),
        )
        logger.info("collection %s created | size=%d distance=%s", collection_name, size, distance)

    def delete_collection(self, collection_name: str | None = None) -> None:
        name = collection_name or self.collection_name
        if name is None:
            return
        try:
            self.client.delete_collection(name)
            logger.info("collection %s deleted", name)
        except Exception as e:
            logger.warning("delete_collection %s failed: %s", name, e)

    def collection_exists(self, collection_name: str | None = None) -> bool:
        return self.client.collection_exists(collection_name or self.collection_name)

    # ── single vector ─────────────────────────────────────────────────

    def add_vector(
        self,
        vector: list[float],
        point_id: int | str | uuid.UUID | None = None,
        payload: dict | None = None,
    ) -> None:
        if self.collection_name is None:
            self.create_collection()
        pid = point_id if point_id is not None else uuid.uuid4()
        point = PointStruct(id=pid, vector=vector, payload=payload or {})
        self.client.upsert(collection_name=self.collection_name, wait=True, points=[point])

    # ── batch ─────────────────────────────────────────────────────────

    def add_vectors(
        self,
        vectors: list[list[float]],
        ids: list[int | str | uuid.UUID] | None = None,
        payloads: list[dict] | None = None,
    ) -> None:
        if self.collection_name is None:
            self.create_collection()
        if ids is None:
            ids = [uuid.uuid4() for _ in range(len(vectors))]
        points = [
            PointStruct(id=pid, vector=vec, payload=pay or {})
            for pid, vec, pay in zip(ids, vectors, payloads or [{}] * len(vectors))
        ]
        self.client.upsert(collection_name=self.collection_name, wait=True, points=points)
        logger.info("added %d vectors to %s", len(points), self.collection_name)

    # ── query ─────────────────────────────────────────────────────────

    def query_vector(
        self,
        query: list[float],
        top_k: int = 3,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        if self.collection_name is None:
            logger.warning("no collection selected for query")
            return []
        kwargs = {"score_threshold": score_threshold} if score_threshold is not None else {}
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query,
            with_payload=True,
            limit=top_k,
            **kwargs,
        )
        return [
            {
                "id": str(p.id),
                "score": p.score,
                "payload": p.payload or {},
            }
            for p in result.points
        ]

    def count_vectors(self) -> int:
        if self.collection_name is None:
            return 0
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0
