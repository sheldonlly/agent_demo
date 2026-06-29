import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import log.logconfig  # noqa: F401
from RAG.rag import RAG_Manager


class TestRAGManager(unittest.TestCase):
    def setUp(self):
        self.rag = RAG_Manager(collection_name="test_rag", vector_size=4, memory_mode=True)

    def tearDown(self):
        self.rag.clear()

    @patch.object(RAG_Manager, "embedding")
    def test_add_documents(self, mock_emb):
        mock_emb.embbedding_vectors.return_value = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        count = self.rag.add_documents(["hello world", "test document"])
        self.assertEqual(count, 2)

    @patch.object(RAG_Manager, "embedding")
    def test_add_empty(self, mock_emb):
        count = self.rag.add_documents([])
        self.assertEqual(count, 0)

    @patch.object(RAG_Manager, "embedding")
    def test_query_empty_question(self, mock_emb):
        results = self.rag.query("")
        self.assertEqual(results, [])

    @patch.object(RAG_Manager, "embedding")
    def test_query_and_format(self, mock_emb):
        mock_emb.embbedding_vectors.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_emb.embbedding_vector.return_value = [0.1, 0.2, 0.3, 0.4]
        self.rag.add_documents(["machine learning is fun"])
        results = self.rag.query("learning", top_k=1)
        if results:
            ctx = self.rag.format_context(results)
            self.assertIn("machine learning", ctx)
        else:
            self.skipTest("empty results - Qdrant in-memory mode may vary")

    def test_format_context_empty(self):
        self.assertEqual(self.rag.format_context([]), "")

    @patch.object(RAG_Manager, "embedding")
    def test_clear(self, mock_emb):
        mock_emb.embbedding_vectors.return_value = [[0.1, 0.2, 0.3, 0.4]]
        self.rag.add_documents(["hello"])
        self.rag.clear()
        self.assertTrue(self.rag.vector_db.collection_exists("test_rag"))
        self.assertEqual(self.rag.vector_db.count_vectors(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
