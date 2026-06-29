import sys
import unittest
from pathlib import Path

_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import log.logconfig  # noqa: F401
from qdrant.qdrantClient import QdrantManager


class TestQdrantManager(unittest.TestCase):
    def setUp(self):
        self.qdrant = QdrantManager(memory_mode=True)
        self.collection = "test_collection"

    def tearDown(self):
        try:
            self.qdrant.delete_collection(self.collection)
        except Exception:
            pass

    def test_is_connected(self):
        self.assertTrue(self.qdrant.is_connected())

    def test_create_and_exists(self):
        self.qdrant.create_collection(self.collection, size=4)
        self.assertTrue(self.qdrant.collection_exists(self.collection))

    def test_double_create(self):
        self.qdrant.create_collection(self.collection, size=4)
        self.qdrant.create_collection(self.collection, size=4)
        self.assertTrue(self.qdrant.collection_exists(self.collection))

    def test_add_and_count(self):
        self.qdrant.create_collection(self.collection, size=4)
        self.qdrant.add_vector([0.1, 0.2, 0.3, 0.4], payload={"content": "hello"})
        self.assertEqual(self.qdrant.count_vectors(), 1)

    def test_add_vectors_batch(self):
        self.qdrant.create_collection(self.collection, size=4)
        vectors = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        payloads = [{"content": "a"}, {"content": "b"}]
        self.qdrant.add_vectors(vectors, payloads=payloads)
        self.assertEqual(self.qdrant.count_vectors(), 2)

    def test_query(self):
        self.qdrant.create_collection(self.collection, size=4)
        self.qdrant.add_vector([1.0, 0.0, 0.0, 0.0], payload={"content": "alpha"})
        self.qdrant.add_vector([0.0, 1.0, 0.0, 0.0], payload={"content": "beta"})
        results = self.qdrant.query_vector([1.0, 0.0, 0.0, 0.0], top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["payload"]["content"], "alpha")

    def test_query_empty_collection(self):
        self.qdrant.collection_name = None
        results = self.qdrant.query_vector([0.1, 0.2])
        self.assertEqual(results, [])

    def test_delete_collection(self):
        self.qdrant.create_collection(self.collection, size=4)
        self.qdrant.delete_collection(self.collection)
        self.assertFalse(self.qdrant.collection_exists(self.collection))

    def test_count_empty(self):
        self.assertEqual(self.qdrant.count_vectors(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
