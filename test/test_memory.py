import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import log.logconfig  # noqa: F401
from memory.memory import BaseMemory, MemoryItem, WorkMemory, SemanticMemory, EpisodicMemory, PerceptualMemory, DATA_DIR


class TestMemoryItem(unittest.TestCase):
    def test_create_default(self):
        item = MemoryItem(content="hello")
        self.assertEqual(item.content, "hello")
        self.assertIsNotNone(item.timestamp)
        self.assertIsInstance(item.metadata, dict)

    def test_create_with_metadata(self):
        item = MemoryItem(content="hello", metadata={"source": "test"})
        self.assertEqual(item.metadata["source"], "test")


class TestBaseMemory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.orig_dir = DATA_DIR
        import memory.memory
        memory.memory.DATA_DIR = self.tmp

    def tearDown(self):
        import memory.memory
        memory.memory.DATA_DIR = self.orig_dir

    def _make_one(self, name="test_mem"):
        return BaseMemory(name)

    def test_add_and_count(self):
        m = self._make_one()
        self.assertEqual(m.count(), 0)
        m.add("hello world")
        self.assertEqual(m.count(), 1)
        m.add("foo bar")
        self.assertEqual(m.count(), 2)

    def test_refuse_empty(self):
        m = self._make_one()
        m.add("")
        self.assertEqual(m.count(), 0)
        m.add("   ")
        self.assertEqual(m.count(), 0)

    def test_search(self):
        m = self._make_one()
        m.add("the quick brown fox")
        m.add("jumps over the lazy dog")
        results = m.search("fox")
        self.assertEqual(len(results), 1)
        self.assertIn("fox", results[0].content)

    def test_search_limit(self):
        m = self._make_one()
        for i in range(10):
            m.add(f"item number {i}")
        results = m.search("item", limit=3)
        self.assertLessEqual(len(results), 3)

    def test_get_all(self):
        m = self._make_one()
        m.add("a")
        m.add("b")
        all_items = m.get_all()
        self.assertEqual(len(all_items), 2)

    def test_clear(self):
        m = self._make_one()
        m.add("data")
        m.clear()
        self.assertEqual(m.count(), 0)

    def test_persistence(self):
        m = self._make_one("persist_test")
        m.add("persist me")
        path = os.path.join(self.tmp, "persist_test.json")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["content"], "persist me")

    def test_load_on_init(self):
        path = os.path.join(self.tmp, "load_test.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{"content": "saved", "timestamp": "now", "metadata": {}}], f)
        m = BaseMemory("load_test")
        self.assertEqual(m.count(), 1)
        self.assertEqual(m.get_all()[0].content, "saved")


class TestConcreteMemories(unittest.TestCase):
    def test_work_memory_context(self):
        wm = WorkMemory()
        wm.clear()
        self.assertEqual(wm.get_context(), "")
        wm.add("first message")
        wm.add("second message")
        ctx = wm.get_context(limit=1)
        self.assertIn("second", ctx)

    def test_semantic_memory(self):
        sm = SemanticMemory()
        sm.clear()
        sm.add("Paris is the capital of France", {"type": "fact"})
        self.assertEqual(sm.count(), 1)

    def test_episodic_memory(self):
        em = EpisodicMemory()
        em.clear()
        em.add("User asked about weather", {"type": "conversation"})
        self.assertEqual(em.count(), 1)

    def test_perceptual_memory(self):
        pm = PerceptualMemory()
        pm.clear()
        pm.add("observed system alert: disk usage 90%")
        self.assertEqual(pm.count(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
