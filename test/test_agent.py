import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import log.logconfig  # noqa: F401
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from agent.agent import (
    ReActAgent,
    ReflectionAgent,
    PlanAndSolveAgent,
    _parse_steps,
    _build_step_context,
    _extract_score,
)


def _make_mock_tool(name: str, result: str = "mock_result") -> BaseTool:
    tool = MagicMock(spec=BaseTool)
    tool.name = name
    tool.invoke.return_value = result
    return tool


def _make_mock_llm(responses: list | None = None):
    """Create a mock LLM that returns predefined responses in sequence."""
    llm = MagicMock()
    if responses:
        llm.invoke.side_effect = responses
    else:
        llm.invoke.return_value = AIMessage(content="mock response")
    return llm


class TestParseSteps(unittest.TestCase):
    def test_parse_numbered(self):
        text = "1. First step\n2. Second step\n3. Third step"
        steps = _parse_steps(text)
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0], "First step")

    def test_parse_dashed(self):
        text = "- Step A\n- Step B"
        steps = _parse_steps(text)
        self.assertEqual(len(steps), 2)

    def test_parse_step_keyword(self):
        text = "Step 1: Do this\nStep 2: Do that"
        steps = _parse_steps(text)
        self.assertEqual(len(steps), 2)

    def test_parse_empty(self):
        self.assertEqual(_parse_steps(""), [])
        self.assertEqual(_parse_steps("  \n  "), [])

    def test_parse_chinese_numbered(self):
        text = "1．第一步\n2．第二步"
        steps = _parse_steps(text)
        self.assertEqual(len(steps), 2)

    def test_parse_mixed(self):
        text = "Plan:\n1. Research\n- Implement\nStep 3: Test"
        steps = _parse_steps(text)
        self.assertEqual(len(steps), 3)


class TestBuildStepContext(unittest.TestCase):
    def test_empty_results(self):
        self.assertEqual(_build_step_context(["a", "b"], []), "")

    def test_with_results(self):
        ctx = _build_step_context(["Step1", "Step2"], ["Result1", "Result2"])
        self.assertIn("Step1", ctx)
        self.assertIn("Result1", ctx)
        self.assertIn("Step2", ctx)
        self.assertIn("Result2", ctx)

    def test_partial_results(self):
        ctx = _build_step_context(["A", "B", "C"], ["R1", "R2"])
        self.assertNotIn("C", ctx)
        self.assertIn("R1", ctx)  # R1 is the result for step A


class TestExtractScore(unittest.TestCase):
    def test_chinese_score(self):
        self.assertEqual(_extract_score("评分：8/10"), "8")

    def test_english_score(self):
        self.assertEqual(_extract_score("Score: 9.5"), "9.5")

    def test_no_score(self):
        self.assertEqual(_extract_score("no score here"), "N/A")

    def test_colon_variants(self):
        self.assertEqual(_extract_score("得分: 7"), "7")
        self.assertEqual(_extract_score("分数：6"), "6")


class TestReActAgent(unittest.TestCase):
    def setUp(self):
        self.mock_llm = _make_mock_llm()
        self.tool = _make_mock_tool("get_weather", "sunny, 25C")
        self.agent = ReActAgent(
            name="TestReAct",
            model=self.mock_llm,
            tools=[self.tool],
        )

    def test_init(self):
        self.assertEqual(self.agent.name, "TestReAct")
        self.assertEqual(len(self.agent.tools), 1)
        self.assertIsNotNone(self.agent._bound)

    def test_init_state(self):
        state = self.agent._init_state("hello")
        self.assertIn("messages", state)
        self.assertIn("iteration", state)
        self.assertEqual(len(state["messages"]), 1)
        self.assertIsInstance(state["messages"][0], HumanMessage)

    def test_extract_answer_from_aimessage(self):
        state = {"messages": [AIMessage(content="final answer")]}
        self.assertEqual(self.agent._extract_answer(state), "final answer")

    def test_extract_answer_empty(self):
        self.assertEqual(self.agent._extract_answer({}), "No response.")

    def test_extract_answer_fallback(self):
        state = {"messages": [HumanMessage(content="hi")]}
        answer = self.agent._extract_answer(state)
        self.assertIn("hi", answer)

    def test_route_continue(self):
        msg = MagicMock(spec=AIMessage)
        msg.tool_calls = [{"name": "get_weather", "args": {}, "id": "1"}]
        state = {"messages": [msg], "iteration": 0}
        route = self.agent._route(state)
        self.assertEqual(route, "continue")

    def test_route_end_no_calls(self):
        msg = AIMessage(content="final")
        state = {"messages": [msg], "iteration": 0}
        route = self.agent._route(state)
        self.assertEqual(route, "end")

    def test_route_end_iteration_limit(self):
        msg = MagicMock(spec=AIMessage)
        msg.tool_calls = [{"name": "get_weather", "args": {}, "id": "1"}]
        state = {"messages": [msg], "iteration": 999}
        route = self.agent._route(state)
        self.assertEqual(route, "end")

    def test_call_agent_llm_error(self):
        self.agent._bound.invoke.side_effect = RuntimeError("API down")
        state = {"messages": [HumanMessage(content="hi")], "iteration": 0}
        result = self.agent._call_agent(state)
        self.assertIn("LLM error", result["messages"][-1].content)

    def test_call_agent_success(self):
        self.mock_llm.invoke.return_value = AIMessage(content="hello")
        state = {"messages": [HumanMessage(content="hi")], "iteration": 0}
        result = self.agent._call_agent(state)
        self.assertEqual(result["iteration"], 1)
        self.assertEqual(len(result["messages"]), 2)

    def test_call_agent_unexpected_type(self):
        self.agent._bound.invoke.return_value = "unexpected string"
        state = {"messages": [HumanMessage(content="hi")], "iteration": 0}
        result = self.agent._call_agent(state)
        self.assertIn("unexpected", result["messages"][-1].content)

    def test_execute_tools_unknown_tool(self):
        msg = MagicMock(spec=AIMessage)
        msg.tool_calls = [{"name": "nonexistent", "args": {}, "id": "1"}]
        state = {"messages": [HumanMessage(content="hi"), msg]}
        result = self.agent._execute_tools(state)
        self.assertIn("Unknown tool", result["messages"][-1].content)

    def test_execute_tools_success(self):
        msg = MagicMock(spec=AIMessage)
        msg.tool_calls = [{"name": "get_weather", "args": {"city": "Beijing"}, "id": "1"}]
        state = {"messages": [HumanMessage(content="hi"), msg]}
        result = self.agent._execute_tools(state)
        self.assertIsInstance(result["messages"][-1], ToolMessage)
        self.assertIn("sunny", result["messages"][-1].content)

    def test_execute_tools_failure(self):
        self.tool.invoke.side_effect = RuntimeError("tool crash")
        msg = MagicMock(spec=AIMessage)
        msg.tool_calls = [{"name": "get_weather", "args": {"city": "Beijing"}, "id": "1"}]
        state = {"messages": [HumanMessage(content="hi"), msg]}
        result = self.agent._execute_tools(state)
        self.assertIn("Error", result["messages"][-1].content)

    @patch.object(ReActAgent, "_call_agent")
    @patch.object(ReActAgent, "_execute_tools")
    def test_run_graph(self, mock_tools, mock_agent):
        mock_agent.return_value = {
            "messages": [HumanMessage(content="q"), AIMessage(content="a")],
            "iteration": 1,
        }
        mock_tools.return_value = {
            "messages": [HumanMessage(content="q"), AIMessage(content="a")],
        }
        result = self.agent.run("test query")
        self.assertIsInstance(result, str)

    def test_run_empty_query(self):
        result = self.agent.run("")
        self.assertIsNotNone(result)

    def test_pre_process_blocked(self):
        mock_mw = MagicMock()
        mock_mw.pre_process.return_value = "Blocked by policy"
        self.agent.middleware = mock_mw
        result = self.agent.run("dangerous query")
        self.assertEqual(result, "Blocked by policy")


class TestReflectionAgent(unittest.TestCase):
    def setUp(self):
        self.mock_llm = _make_mock_llm()
        self.agent = ReflectionAgent(
            name="TestReflection",
            model=self.mock_llm,
            max_iterations=2,
        )

    def test_init_state(self):
        state = self.agent._init_state("test")
        self.assertEqual(state["draft"], "")
        self.assertEqual(state["iteration"], 0)

    def test_extract_answer_from_draft(self):
        state = {"draft": "refined answer"}
        self.assertEqual(self.agent._extract_answer(state), "refined answer")

    def test_extract_answer_fallback(self):
        state = {"draft": "", "messages": [AIMessage(content="fallback")]}
        self.assertEqual(self.agent._extract_answer(state), "fallback")

    def test_generate_draft_success(self):
        self.mock_llm.invoke.return_value = AIMessage(content="initial draft")
        state = {"messages": [HumanMessage(content="explain AI")], "draft": "", "iteration": 0}
        result = self.agent._generate_draft(state)
        self.assertEqual(result["draft"], "initial draft")
        self.assertEqual(result["iteration"], 1)

    def test_generate_draft_error(self):
        self.mock_llm.invoke.side_effect = RuntimeError("LLM error")
        state = {"messages": [HumanMessage(content="explain AI")], "draft": "", "iteration": 0}
        result = self.agent._generate_draft(state)
        self.assertIn("Draft error", result["messages"][-1].content)
        self.assertEqual(result["draft"], "")

    def test_reflect_empty_draft(self):
        state = {"messages": [HumanMessage(content="q")], "draft": "", "iteration": 0}
        result = self.agent._reflect(state)
        self.assertIn("PASS", result["messages"][-1].content)

    def test_reflect_success(self):
        self.mock_llm.invoke.return_value = AIMessage(content="评分：8/10\n结论：PASS")
        state = {"messages": [HumanMessage(content="q")], "draft": "some answer", "iteration": 1}
        result = self.agent._reflect(state)
        self.assertIn("PASS", result["messages"][-1].content.upper())

    def test_reflect_error(self):
        self.mock_llm.invoke.side_effect = RuntimeError("reflect error")
        state = {"messages": [HumanMessage(content="q")], "draft": "some answer", "iteration": 1}
        result = self.agent._reflect(state)
        self.assertIn("Reflect error", result["messages"][-1].content)

    def test_revise_empty_draft(self):
        state = {"messages": [], "draft": "", "iteration": 0}
        result = self.agent._revise(state)
        self.assertEqual(result["draft"], "")

    def test_revise_success(self):
        self.mock_llm.invoke.return_value = AIMessage(content="revised answer")
        state = {
            "messages": [
                HumanMessage(content="q"),
                AIMessage(content="评分：5/10\n结论：FAIL\n理由：不够详细"),
            ],
            "draft": "original answer",
            "iteration": 1,
        }
        result = self.agent._revise(state)
        self.assertEqual(result["draft"], "revised answer")

    def test_revise_error(self):
        self.mock_llm.invoke.side_effect = RuntimeError("revise error")
        state = {
            "messages": [HumanMessage(content="q")],
            "draft": "original answer",
            "iteration": 1,
        }
        result = self.agent._revise(state)
        self.assertIn("Revise error", result["messages"][-1].content)

    def test_route_revise_on_fail(self):
        msg = AIMessage(content="评分：3/10\n结论：FAIL")
        state = {"messages": [msg], "draft": "draft", "iteration": 0}
        self.assertEqual(self.agent._route(state), "revise")

    def test_route_end_on_pass(self):
        msg = AIMessage(content="评分：9/10\n结论：PASS")
        state = {"messages": [msg], "draft": "draft", "iteration": 0}
        self.assertEqual(self.agent._route(state), "end")

    def test_route_end_on_iteration_exhausted(self):
        msg = AIMessage(content="评分：3/10\n结论：FAIL")
        state = {"messages": [msg], "draft": "draft", "iteration": 999}
        self.assertEqual(self.agent._route(state), "end")


class TestPlanAndSolveAgent(unittest.TestCase):
    def setUp(self):
        self.mock_llm = _make_mock_llm()
        self.agent = PlanAndSolveAgent(
            name="TestPlanSolve",
            model=self.mock_llm,
            max_steps=3,
        )

    def test_init_state(self):
        state = self.agent._init_state("plan a trip")
        self.assertEqual(state["plan"], [])
        self.assertEqual(state["plan_text"], "")
        self.assertEqual(state["current_step"], 0)
        self.assertEqual(state["step_results"], [])

    def test_do_plan_success(self):
        self.mock_llm.invoke.return_value = AIMessage(content="PLAN:\n1. Research\n2. Book\n3. Pack")
        state = {"messages": [HumanMessage(content="plan trip")], "plan": [], "plan_text": "", "current_step": 0, "step_results": [], "iteration": 0}
        result = self.agent._do_plan(state)
        self.assertEqual(len(result["plan"]), 3)
        self.assertEqual(result["plan"][0], "Research")

    def test_do_plan_empty(self):
        self.mock_llm.invoke.return_value = AIMessage(content="No plan needed")
        state = {"messages": [HumanMessage(content="hi")], "plan": [], "plan_text": "", "current_step": 0, "step_results": [], "iteration": 0}
        result = self.agent._do_plan(state)
        self.assertGreaterEqual(len(result["plan"]), 1)

    def test_do_plan_error(self):
        self.mock_llm.invoke.side_effect = RuntimeError("plan error")
        state = {"messages": [HumanMessage(content="hi")], "plan": [], "plan_text": "", "current_step": 0, "step_results": [], "iteration": 0}
        result = self.agent._do_plan(state)
        self.assertIn("Plan error", result["messages"][-1].content)

    def test_do_plan_truncation(self):
        self.mock_llm.invoke.return_value = AIMessage(content="PLAN:\n1. A\n2. B\n3. C\n4. D\n5. E")
        state = {"messages": [HumanMessage(content="hi")], "plan": [], "plan_text": "", "current_step": 0, "step_results": [], "iteration": 0}
        result = self.agent._do_plan(state)
        self.assertLessEqual(len(result["plan"]), self.agent.max_steps)

    def test_do_solve_past_end(self):
        state = {"messages": [HumanMessage(content="q")], "plan": ["A", "B"], "plan_text": "", "current_step": 999, "step_results": [], "iteration": 0}
        result = self.agent._do_solve(state)
        self.assertIsNotNone(result)

    def test_do_solve_success(self):
        self.mock_llm.invoke.return_value = AIMessage(content="Step result")
        state = {"messages": [HumanMessage(content="q")], "plan": ["Research", "Book"], "plan_text": "PLAN:\n1. Research\n2. Book", "current_step": 0, "step_results": [], "iteration": 0}
        result = self.agent._do_solve(state)
        self.assertEqual(len(result["step_results"]), 1)
        self.assertEqual(result["current_step"], 1)
        self.assertIn("Step result", result["step_results"][0])

    def test_do_solve_error(self):
        self.mock_llm.invoke.side_effect = RuntimeError("solve error")
        state = {"messages": [HumanMessage(content="q")], "plan": ["Research"], "plan_text": "PLAN:\n1. Research", "current_step": 0, "step_results": [], "iteration": 0}
        result = self.agent._do_solve(state)
        self.assertIn("error", result["step_results"][0].lower())

    def test_do_refine_empty(self):
        state = {"messages": [HumanMessage(content="q")], "step_results": []}
        result = self.agent._do_refine(state)
        self.assertIsNotNone(result)

    def test_do_refine_success(self):
        self.mock_llm.invoke.return_value = AIMessage(content="Refined summary")
        state = {"messages": [HumanMessage(content="q")], "step_results": ["Result A", "Result B"]}
        result = self.agent._do_refine(state)
        self.assertIsNotNone(result)

    def test_do_refine_error_fallback(self):
        self.mock_llm.invoke.side_effect = RuntimeError("refine error")
        state = {"messages": [HumanMessage(content="q")], "step_results": ["Result A"]}
        result = self.agent._do_refine(state)
        self.assertIn("Result A", str(result["messages"][-1].content))

    def test_after_solve_more_steps(self):
        state = {"plan": ["A", "B", "C"], "current_step": 0}
        self.assertEqual(self.agent._after_solve(state), "solve")

    def test_after_solve_done(self):
        state = {"plan": ["A", "B"], "current_step": 2}
        self.assertEqual(self.agent._after_solve(state), "refine")


class TestAgentIntegration(unittest.TestCase):
    def test_import_all_agents(self):
        from agent.agent import BaseAgent, ReActAgent, ReflectionAgent, PlanAndSolveAgent
        self.assertTrue(issubclass(ReActAgent, BaseAgent))
        self.assertTrue(issubclass(ReflectionAgent, BaseAgent))
        self.assertTrue(issubclass(PlanAndSolveAgent, BaseAgent))

    def test_base_agent_abstract(self):
        from agent.agent import BaseAgent
        with self.assertRaises(TypeError):
            BaseAgent("test", _make_mock_llm())

    def test_default_middleware_created(self):
        llm = _make_mock_llm()
        agent = ReActAgent(name="Test", model=llm)
        self.assertIsNotNone(agent.middleware)


if __name__ == "__main__":
    unittest.main(verbosity=2)
