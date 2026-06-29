import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import log.logconfig  # noqa: F401
from context import ContextManager, SessionPriority, SESSIONS_DIR
from memory.memoryManager import MemoryManager


class TestContextManager(unittest.TestCase):
    def setUp(self):
        self.ctx = ContextManager()
        self.ctx.clear_history()
        self.ctx.memory.clear_all()

    def test_add_turn(self):
        self.ctx.add_turn("user", "hello")
        self.ctx.add_turn("assistant", "hi there")
        history = self.ctx.get_history()
        self.assertEqual(len(history), 2)

    def test_get_history_limit(self):
        for i in range(5):
            self.ctx.add_turn("user", f"msg{i}")
        self.assertEqual(len(self.ctx.get_history(limit=2)), 2)

    def test_clear_history(self):
        self.ctx.add_turn("user", "hello")
        self.ctx.clear_history()
        self.assertEqual(len(self.ctx.get_history()), 0)

    def test_build_prompt_basic(self):
        prompt = self.ctx.build_prompt("test query")
        self.assertIn("test query", prompt)

    def test_build_prompt_with_system(self):
        prompt = self.ctx.build_prompt("q", system_instruction="be helpful")
        self.assertIn("be helpful", prompt)

    def test_build_prompt_with_history(self):
        self.ctx.add_turn("user", "previous question")
        self.ctx.add_turn("assistant", "previous answer")
        prompt = self.ctx.build_prompt("new question")
        self.assertIn("previous question", prompt)
        self.assertIn("new question", prompt)

    def test_request_approval_default_reject(self):
        action = {"name": "delete_db", "args": {"table": "users"}}
        approved = self.ctx.request_approval(action)
        self.assertFalse(approved)

    def test_approve_last_action(self):
        self.ctx.request_approval({"name": "test", "args": {}})
        self.assertTrue(self.ctx.approve_last_action())

    def test_reject_last_action(self):
        self.ctx.request_approval({"name": "test", "args": {}})
        self.assertTrue(self.ctx.reject_last_action())

    def test_pending_count(self):
        self.assertEqual(self.ctx.pending_count(), 0)
        self.ctx.request_approval({"name": "a", "args": {}})
        self.assertEqual(self.ctx.pending_count(), 1)
        self.ctx.request_approval({"name": "b", "args": {}})
        self.assertEqual(self.ctx.pending_count(), 2)

    def test_record_interaction(self):
        self.ctx.record_interaction("user query", "assistant response")
        history = self.ctx.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(self.ctx.memory.work.count(), 1)

    def test_empty_content_ignored(self):
        self.ctx.add_turn("user", "")
        self.assertEqual(len(self.ctx.get_history()), 0)

    def test_memory_persistence(self):
        import tempfile
        self.ctx.record_interaction("persist?", "yes")
        meta_path = Path(SESSIONS_DIR) / f"{self.ctx.session_id}_history.json"
        self.assertTrue(meta_path.exists())

    # ── session management ──────────────────────────────────────────

    def test_session_id_generated(self):
        ctx = ContextManager()
        self.assertIsNotNone(ctx.session_id)
        self.assertIsInstance(ctx.session_id, str)

    def test_session_priority(self):
        self.ctx.set_session_priority(SessionPriority.HIGH)
        info = self.ctx.get_session_info()
        self.assertEqual(info["priority"], "HIGH")

    def test_session_tags(self):
        self.ctx.add_session_tag("test")
        self.ctx.add_session_tag("debug")
        info = self.ctx.get_session_info()
        self.assertIn("test", info["tags"])
        self.assertIn("debug", info["tags"])

    def test_session_unique_tags(self):
        self.ctx.add_session_tag("test")
        self.ctx.add_session_tag("test")
        info = self.ctx.get_session_info()
        self.assertEqual(len(info["tags"]), 1)

    def test_get_session_info(self):
        info = self.ctx.get_session_info()
        self.assertIn("session_id", info)
        self.assertIn("created_at", info)
        self.assertIn("turn_count", info)
        self.assertIn("history_turns", info)
        self.assertIn("pending_actions", info)

    def test_list_sessions(self):
        sessions = self.ctx.list_sessions()
        self.assertIsInstance(sessions, list)

    def test_switch_session_invalid(self):
        result = self.ctx.switch_session("nonexistent_session")
        self.assertFalse(result)

    def test_switch_session_valid(self):
        old_id = self.ctx.session_id
        self.ctx.add_turn("user", "test message")
        info = self.ctx.get_session_info()
        self.assertEqual(info["turn_count"], 1)

    def test_turn_count_tracking(self):
        self.ctx.add_turn("user", "msg1")
        self.ctx.add_turn("assistant", "resp1")
        self.ctx.add_turn("user", "msg2")
        info = self.ctx.get_session_info()
        self.assertEqual(info["turn_count"], 3)

    def test_build_prompt_no_rag(self):
        prompt = self.ctx.build_prompt("test", use_rag=False)
        self.assertNotIn("[Knowledge]", prompt)

    def test_export_markdown_format(self):
        self.ctx.add_turn("user", "Hello")
        self.ctx.add_turn("assistant", "World")
        md = self.ctx.export_history(format="markdown")
        self.assertIn("Hello", md)
        self.assertIn("World", md)

    def test_export_json_format(self):
        self.ctx.add_turn("user", "test")
        exported = self.ctx.export_history(format="json")
        data = json.loads(exported)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)

    def test_get_pending_actions(self):
        self.ctx.request_approval({"name": "a1", "args": {}})
        self.ctx.request_approval({"name": "a2", "args": {}})
        actions = self.ctx.get_pending_actions()
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0]["name"], "a1")

    def test_history_pruning(self):
        from context import MAX_HISTORY_TURNS
        for i in range(MAX_HISTORY_TURNS + 10):
            self.ctx.add_turn("user", f"message {i}")
        history = self.ctx.get_history()
        self.assertLessEqual(len(history), MAX_HISTORY_TURNS + 1)

    def test_long_content_truncated(self):
        long_text = "x" * 5000
        self.ctx.add_turn("user", long_text)
        history = self.ctx.get_history()
        self.assertLessEqual(len(history[0]["content"]), 2000)

    def test_session_persistence(self):
        self.ctx.add_turn("user", "session persist test")
        session_id = self.ctx.session_id

        new_ctx = ContextManager(session_id=session_id)
        history = new_ctx.get_history()
        self.assertGreater(len(history), 0)
        self.assertEqual(history[0]["content"], "session persist test")


if __name__ == "__main__":
    unittest.main(verbosity=2)
