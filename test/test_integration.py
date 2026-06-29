import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import log.logconfig  # noqa: F401
from langchain_core.messages import AIMessage, HumanMessage

from context import ContextManager
from memory.memoryManager import MemoryManager
from middleware.middleware import Middleware
from RAG.rag import RAG_Manager


class TestContextWithMemory(unittest.TestCase):
    def setUp(self):
        self.memory = MemoryManager()
        self.memory.clear_all()
        self.ctx = ContextManager(memory=self.memory)
        self.ctx.clear_history()

    def test_record_and_retrieve(self):
        self.ctx.record_interaction("What is Python?", "Python is a language")
        history = self.ctx.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(self.memory.work.count(), 1)

    def test_build_prompt_with_memory(self):
        self.memory.record_fact("Python was created by Guido van Rossum")
        prompt = self.ctx.build_prompt("Who created Python?")
        self.assertIn("Python", prompt)
        self.assertNotIn("[Knowledge]", prompt)  # no RAG

    def test_build_prompt_with_history(self):
        self.ctx.add_turn("user", "Hello")
        self.ctx.add_turn("assistant", "Hi there")
        prompt = self.ctx.build_prompt("How are you?")
        self.assertIn("Hello", prompt)
        self.assertIn("How are you?", prompt)

    def test_multiple_sessions_isolation(self):
        ctx1 = ContextManager(memory=self.memory)
        ctx1.clear_history()
        ctx1.add_turn("user", "msg in session 1")

        ctx2 = ContextManager(memory=self.memory)
        ctx2.clear_history()
        ctx2.add_turn("user", "msg in session 2")

        self.assertEqual(len(ctx1.get_history()), 1)
        self.assertEqual(len(ctx2.get_history()), 1)

    def test_export_markdown(self):
        self.ctx.add_turn("user", "Hello")
        self.ctx.add_turn("assistant", "World")
        md = self.ctx.export_history(format="markdown")
        self.assertIn("Hello", md)
        self.assertIn("World", md)

    def test_export_json(self):
        self.ctx.add_turn("user", "test")
        exported = self.ctx.export_history(format="json")
        self.assertIn("test", exported)

    def test_session_info(self):
        info = self.ctx.get_session_info()
        self.assertIn("session_id", info)
        self.assertIn("turn_count", info)
        self.assertIn("priority", info)

    def test_pending_actions_list(self):
        self.ctx.request_approval({"name": "action1", "args": {}})
        self.ctx.request_approval({"name": "action2", "args": {}})
        pending = self.ctx.get_pending_actions()
        self.assertEqual(len(pending), 2)
        self.assertEqual(pending[0]["name"], "action1")

    def test_approve_reject_cycle(self):
        self.ctx.request_approval({"name": "test", "args": {}})
        self.assertTrue(self.ctx.approve_last_action())
        self.assertEqual(self.ctx.pending_count(), 0)

    def test_reject_none(self):
        self.assertFalse(self.ctx.reject_last_action())

    def test_approve_none(self):
        self.assertFalse(self.ctx.approve_last_action())


class TestMiddlewareWithContext(unittest.TestCase):
    def setUp(self):
        self.ctx = ContextManager()
        self.ctx.clear_history()
        self.mw = Middleware(self.ctx)

    def test_high_risk_creates_pending_action(self):
        tool_calls = [{"name": "bash", "args": {"cmd": "rm -rf /"}, "id": "1"}]
        self.mw.pre_process("test", tool_calls)
        self.assertGreaterEqual(self.ctx.pending_count(), 1)

    def test_safe_tool_no_pending(self):
        tool_calls = [{"name": "get_weather", "args": {"city": "Beijing"}, "id": "1"}]
        self.mw.pre_process("test", tool_calls)
        self.assertEqual(self.ctx.pending_count(), 0)

    def test_hook_integration(self):
        results = []

        def tracking_hook(q, tc):
            results.append((q, tc))

        self.mw.register_pre_hook(tracking_hook)
        self.mw.pre_process("hello", [{"name": "test", "args": {}, "id": "1"}])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "hello")


@patch("RAG.rag.EMBEDDING_LLM")
class TestRAGWithContext(unittest.TestCase):
    def setUp(self):
        self.rag = RAG_Manager(collection_name="test_int_rag", vector_size=4, memory_mode=True)
        self.ctx = ContextManager(rag=self.rag)

    def tearDown(self):
        self.rag.clear()

    def test_rag_query_in_build_prompt(self, mock_emb):
        mock_emb_instance = mock_emb.return_value
        mock_emb_instance.embbedding_vectors.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_emb_instance.embbedding_vector.return_value = [0.1, 0.2, 0.3, 0.4]

        self.rag.add_documents(["Eiffel Tower is in Paris"])
        prompt = self.ctx.build_prompt("Where is Eiffel Tower?", use_rag=True)
        self.assertIn("[Knowledge]", prompt)

    def test_rag_disabled(self, mock_emb):
        prompt = self.ctx.build_prompt("Where is Eiffel Tower?", use_rag=False)
        self.assertNotIn("[Knowledge]", prompt)


class TestMemoryManagerIntegration(unittest.TestCase):
    def setUp(self):
        self.mgr = MemoryManager()
        self.mgr.clear_all()

    def test_full_lifecycle(self):
        self.mgr.record_interaction("Q1", "A1")
        self.mgr.record_fact("Fact 1")
        self.mgr.record_observation("Obs 1")

        stats = self.mgr.stats()
        self.assertEqual(stats["work"], 1)
        self.assertEqual(stats["semantic"], 1)
        self.assertEqual(stats["perceptual"], 1)
        self.assertEqual(stats["episodic"], 1)

        results = self.mgr.search_all("Q1")
        self.assertGreater(len(results["work"]), 0)
        self.assertGreater(len(results["episodic"]), 0)

        ctx = self.mgr.get_recent_context()
        self.assertIn("Q1", ctx)

        self.mgr.clear_all()
        self.assertEqual(sum(self.mgr.stats().values()), 0)

    def test_multiple_interactions(self):
        for i in range(5):
            self.mgr.record_interaction(f"Q{i}", f"A{i}")
        self.assertEqual(self.mgr.work.count(), 5)
        self.assertEqual(self.mgr.episodic.count(), 5)

    def test_empty_interaction(self):
        self.mgr.record_interaction("", "")
        self.assertEqual(self.mgr.work.count(), 0)


class TestFullPipelineWithMocks(unittest.TestCase):
    @patch("agent.agent.ReActAgent._call_agent")
    @patch("agent.agent.ReActAgent._execute_tools")
    def test_agent_with_context_and_middleware(self, mock_tools, mock_agent):
        from agent.agent import ReActAgent
        from model.chat_llm import LLM

        mock_tools.return_value = {
            "messages": [HumanMessage(content="q"), AIMessage(content="a")],
        }
        mock_agent.return_value = {
            "messages": [HumanMessage(content="q"), AIMessage(content="final")],
            "iteration": 1,
        }

        llm = MagicMock()
        llm.llm = MagicMock()
        llm.llm.invoke.return_value = AIMessage(content="test")

        memory = MemoryManager()
        ctx = ContextManager(memory=memory)
        mw = Middleware(ctx)

        agent = ReActAgent(
            name="IntegrationTest",
            model=llm.llm,
            tools=[],
            memory=memory,
            context=ctx,
            middleware=mw,
        )

        result = agent.run("test query")
        self.assertIsInstance(result, str)
        self.assertGreater(len(ctx.get_history()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
