import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import log.logconfig  # noqa: F401
from middleware.middleware import Middleware, HIGH_RISK_KEYWORDS


class TestMiddleware(unittest.TestCase):
    def setUp(self):
        self.mock_context = MagicMock()
        self.mw = Middleware(self.mock_context)

    def test_init(self):
        self.assertEqual(len(self.mw._pre_hooks), 0)
        self.assertEqual(len(self.mw._post_hooks), 0)

    def test_register_pre_hook(self):
        hook = lambda q, tc: None
        self.mw.register_pre_hook(hook)
        self.assertEqual(len(self.mw._pre_hooks), 1)

    def test_register_post_hook(self):
        hook = lambda q, r: None
        self.mw.register_post_hook(hook)
        self.assertEqual(len(self.mw._post_hooks), 1)

    # ── high-risk detection ──────────────────────────────────────────

    def test_high_risk_detection_delete(self):
        tool_calls = [{"name": "bash", "args": {"command": "rm -rf /"}, "id": "1"}]
        self.mock_context.request_approval.return_value = False
        result = self.mw._check_high_risk(tool_calls)
        self.assertIsNotNone(result)
        self.assertIn("Blocked", result)

    def test_high_risk_detection_drop(self):
        tool_calls = [{"name": "sql_query", "args": {"query": "DROP TABLE users"}, "id": "1"}]
        self.mock_context.request_approval.return_value = False
        result = self.mw._check_high_risk(tool_calls)
        self.assertIsNotNone(result)

    def test_high_risk_detection_shutdown(self):
        tool_calls = [{"name": "system", "args": {"cmd": "shutdown -h now"}, "id": "1"}]
        self.mock_context.request_approval.return_value = False
        result = self.mw._check_high_risk(tool_calls)
        self.assertIsNotNone(result)

    def test_high_risk_approval_granted(self):
        tool_calls = [{"name": "bash", "args": {"command": "rm -rf /"}, "id": "1"}]
        self.mock_context.request_approval.return_value = True
        result = self.mw._check_high_risk(tool_calls)
        self.assertIsNone(result)

    def test_safe_tools_not_blocked(self):
        tool_calls = [{"name": "get_weather", "args": {"city": "Beijing"}, "id": "1"}]
        result = self.mw._check_high_risk(tool_calls)
        self.assertIsNone(result)

    def test_no_tool_calls(self):
        result = self.mw._check_high_risk(None)
        self.assertIsNone(result)

    def test_empty_tool_calls(self):
        result = self.mw._check_high_risk([])
        self.assertIsNone(result)

    def test_all_high_risk_keywords_defined(self):
        self.assertIn("delete", HIGH_RISK_KEYWORDS)
        self.assertIn("drop ", HIGH_RISK_KEYWORDS)
        self.assertIn("exec(", HIGH_RISK_KEYWORDS)
        self.assertIn("subprocess", HIGH_RISK_KEYWORDS)

    # ── pre_process pipeline ─────────────────────────────────────────

    def test_pre_process_no_hooks(self):
        result = self.mw.pre_process("hello")
        self.assertIsNone(result)

    def test_pre_process_with_hooks(self):
        self.mw.register_pre_hook(lambda q, tc: None)
        self.mw.register_pre_hook(lambda q, tc: "blocked" if "bad" in q else None)
        result = self.mw.pre_process("good query")
        self.assertIsNone(result)
        result = self.mw.pre_process("bad query")
        self.assertEqual(result, "blocked")

    def test_pre_process_hook_exception(self):
        def failing_hook(q, tc):
            raise ValueError("hook error")

        self.mw.register_pre_hook(failing_hook)
        result = self.mw.pre_process("test")
        self.assertIsNone(result)

    def test_pre_process_high_risk_triggers_approval(self):
        tool_calls = [{"name": "run", "args": {"code": "exec('evil')"}, "id": "1"}]
        self.mock_context.request_approval.return_value = False
        result = self.mw.pre_process("test", tool_calls)
        self.assertIsNotNone(result)
        self.mock_context.request_approval.assert_called_once()

    # ── post_process pipeline ────────────────────────────────────────

    def test_post_process_no_hooks(self):
        result = self.mw.post_process("q", "response")
        self.assertEqual(result, "response")

    def test_post_process_with_hooks(self):
        self.mw.register_post_hook(lambda q, r: r.upper() if "important" in q else None)
        result = self.mw.post_process("normal", "hello")
        self.assertEqual(result, "hello")
        result = self.mw.post_process("important", "hello")
        self.assertEqual(result, "HELLO")

    def test_post_process_hook_exception(self):
        def failing_hook(q, r):
            raise ValueError("post error")

        self.mw.register_post_hook(failing_hook)
        result = self.mw.post_process("q", "response")
        self.assertEqual(result, "response")

    def test_post_process_chaining(self):
        self.mw.register_post_hook(lambda q, r: r + " [processed]")
        self.mw.register_post_hook(lambda q, r: r.upper())
        result = self.mw.post_process("q", "hello")
        self.assertEqual(result, "HELLO [PROCESSED]")

    # ── built-in hooks ───────────────────────────────────────────────

    def test_sanitize_output_hook(self):
        result = Middleware.sanitize_output_hook("q", "[LLM error: timeout]")
        self.assertIsNone(result)
        result = Middleware.sanitize_output_hook("q", "normal response")
        self.assertIsNone(result)

    def test_log_interaction_hook(self):
        result = Middleware.log_interaction_hook("query", "response")
        self.assertIsNone(result)

    # ── full pipeline integration ────────────────────────────────────

    def test_pre_process_returns_none_for_safe(self):
        tool_calls = [{"name": "get_weather", "args": {"city": "Beijing"}, "id": "1"}]
        result = self.mw.pre_process("weather?", tool_calls)
        self.assertIsNone(result)

    def test_pre_process_detects_dangerous_command(self):
        tool_calls = [{"name": "execute", "args": {"cmd": "format disk"}, "id": "1"}]
        self.mock_context.request_approval.return_value = False
        result = self.mw.pre_process("danger", tool_calls)
        self.assertIsNotNone(result)
        self.assertIn("Blocked", result)

    def test_context_approval_requested(self):
        tool_calls = [{"name": "delete_file", "args": {"path": "/etc"}, "id": "1"}]
        self.mock_context.request_approval.return_value = False
        self.mw.pre_process("test", tool_calls)
        self.mock_context.request_approval.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
