from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import logging
import log.logconfig

logger = logging.getLogger(__name__)

class QdrantManager:
    def __init__(self,
                 ip: str = "127.0.0.1",
                 port: int = 6333,
                 timeout: int = 30,
                 trust_env: bool = False):
        self.url = f"http://{ip}:{port}"
        self.client = QdrantClient(url=self.url,
                                   timeout=timeout,
                                   **{"trust_env": trust_env})
        logger.info(f"QdrantManager connected to {self.url}")

    def test_is_connected(self):
        collections = self.client.get_collections()
        print(collections)

    def create_collection(self,
                         collection_name: str = "sheldonDemo",
                         size: int = 1024,
                         distance: Distance = Distance.COSINE):
        self.collection_name = collection_name
        if self.client.collection_exists(collection_name):
            logger.warning(f"Collection {collection_name} already exists")
            return
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=size,
                distance=distance
            )
        )

    def add_vector(self,
                   vector: list,
                   id: [int | str | uuid.UUID] | None = None,
                   payload: dict | None = None):
        if self.collection_name is None:
            self.create_collection()
        if id is None:
            id = uuid.uuid4()
        point = PointStruct(
            id=id,
            vector=vector,
            payload=payload
        )
        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=[point])

    def add_vectors(self,
                    vectors: list[list],
                    ids: list[[int | str | uuid.UUID]] | None = None,
                    payloads: list[dict] | None = None):
        if self.collection_name is None:
            self.create_collection()
        if ids is None:
            ids = [uuid.uuid4() for i in range(len(vectors))]
        points = []
        if payloads is None:
            logger.warning(f"payloads is None")
            for id, vector in zip(ids, vectors):
                points.append(PointStruct(id=id, vector=vector))
        else:
            for id, vector, payload in zip(ids, vectors, payloads):
                points.append(PointStruct(id=id, vector=vector, payload=payload))
        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points)

    def query_vector(self, query: list):
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query,
            with_payload=True,
            limit=3
        )

    def delete_collection(self, collection_name: str = None):
        tmp = collection_name or self.collection_name
        if tmp is None:
            return
        self.client.delete_collection(tmp)



if __name__ == '__main__':
    logger.info("正在创建QdrantManager")
    client = QdrantManager()
    client.test_is_connected()
    client.create_collection()

    s1 = "今天天气很好"
    s2 = "今天天气怎么样"
    s3 = "今天是什么样的天气"
    s4 = "今天温度怎么样"
    s5 = "qdrant是一个向量数据库"
    s6 = "我是一名中国人"
    s = [s1, s2, s3, s4, s5, s6]
    payloads = [{"content":t} for t in s]

    vectors = []
    from model.embedding_llm import EMBEDDING_LLM
    embbeding_model = EMBEDDING_LLM()
    vectors = embbeding_model.embbedding_vectors(s)
    client.add_vectors(vectors=vectors, payloads=payloads)

    q = "今天温度有点低"
    q_vector = embbeding_model.embbedding_vector(q)
    search_vector = client.query_vector(q_vector)
    print(search_vector)

    qs = ["请问你是谁", "qdrant是什么东西", "今天天气挺好的"]
    q_vectors = embbeding_model.embbedding_vectors(qs)
    for t in q_vectors:
        ans = client.query_vector(t)
        print(ans)

    client.delete_collection()
