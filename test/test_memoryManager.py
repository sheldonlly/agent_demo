import sys
import unittest
from pathlib import Path

_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import log.logconfig  # noqa: F401
from memory.memoryManager import MemoryManager


class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        self.mgr = MemoryManager()
        self.mgr.clear_all()

    def test_record_interaction(self):
        self.mgr.record_interaction("hello", "world")
        self.assertEqual(self.mgr.work.count(), 1)
        self.assertEqual(self.mgr.episodic.count(), 1)

    def test_record_fact(self):
        self.mgr.record_fact("Earth revolves around the Sun")
        self.assertEqual(self.mgr.semantic.count(), 1)

    def test_record_observation(self):
        self.mgr.record_observation("temp: 25C")
        self.assertEqual(self.mgr.perceptual.count(), 1)

    def test_search_all(self):
        self.mgr.record_fact("Python is a programming language")
        results = self.mgr.search_all("Python")
        self.assertIn("semantic", results)
        self.assertGreater(len(results["semantic"]), 0)

    def test_stats(self):
        self.mgr.record_fact("fact1")
        self.mgr.record_interaction("q", "a")
        stats = self.mgr.stats()
        self.assertEqual(stats["work"], 1)
        self.assertEqual(stats["semantic"], 1)
        self.assertEqual(stats["episodic"], 1)

    def test_clear_all(self):
        self.mgr.record_fact("something")
        self.mgr.clear_all()
        self.assertEqual(sum(self.mgr.stats().values()), 0)

    def test_get_recent_context_empty(self):
        self.assertEqual(self.mgr.get_recent_context(), "")

    def test_get_recent_context(self):
        self.mgr.record_interaction("how are you?", "I'm fine")
        ctx = self.mgr.get_recent_context()
        self.assertIn("how are you?", ctx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
