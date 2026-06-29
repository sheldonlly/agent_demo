import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import log.logconfig  # noqa: F401
from agent.agent import (
    BaseAgent, ReActAgent, ReflectionAgent, PlanAndSolveAgent,
    _parse_steps, _build_step_context, _extract_score,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool


class TestHelpers(unittest.TestCase):
    def test_parse_steps_numbered(self):
        text = "PLAN:\n1. step one\n2. step two\n3. step three"
        steps = _parse_steps(text)
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0], "step one")

    def test_parse_steps_dashed(self):
        text = "PLAN:\n- first\n- second"
        steps = _parse_steps(text)
        self.assertEqual(len(steps), 2)

    def test_parse_steps_empty(self):
        self.assertEqual(_parse_steps(""), [])

    def test_build_step_context_empty(self):
        self.assertEqual(_build_step_context(["a", "b"], []), "")

    def test_build_step_context(self):
        ctx = _build_step_context(["a", "b"], ["result_a"])
        self.assertIn("Step 1", ctx)
        self.assertIn("result_a", ctx)
        self.assertNotIn("Step 2", ctx)

    def test_extract_score_found(self):
        self.assertEqual(_extract_score("评分：8/10"), "8")
        self.assertEqual(_extract_score("分数: 9.5/10"), "9.5")

    def test_extract_score_not_found(self):
        self.assertEqual(_extract_score("no score here"), "N/A")


class TestBaseAgent(unittest.TestCase):
    def test_init_sets_attributes(self):
        model = MagicMock()
        agent = _create_test_agent(model)
        self.assertEqual(agent.name, "TestAgent")
        self.assertIsNotNone(agent.context)
        self.assertIsNotNone(agent.memory)
        self.assertIsNotNone(agent.middleware)

    def test_run_with_mocked_model(self):
        model = MagicMock()
        model.invoke.return_value = AIMessage(content="mocked answer")
        agent = _create_test_agent(model)
        result = agent.run("test query")
        self.assertIn("mocked", result)

    def test_run_handles_error(self):
        model = MagicMock()
        model.invoke.side_effect = Exception("API error")
        agent = _create_test_agent(model)
        result = agent.run("test")
        self.assertIn("error", result.lower())

    def test_run_records_interaction(self):
        model = MagicMock()
        model.invoke.return_value = AIMessage(content="answer")
        agent = _create_test_agent(model)
        agent.run("hello")
        self.assertGreater(agent.context.memory.work.count(), 0)


def _create_test_agent(model):
    class TestAgent(BaseAgent):
        def build_graph(self):
            from langgraph.graph import StateGraph, START, END
            from agent.agent import _BaseState
            builder = StateGraph(_BaseState)
            builder.add_node("call", lambda s: {
                "messages": s["messages"] + [model.invoke(s["messages"])],
                "iteration": s.get("iteration", 0) + 1,
            })
            builder.add_edge(START, "call")
            builder.add_edge("call", END)
            return builder.compile()
    return TestAgent(name="TestAgent", model=model)


class TestReActAgent(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock()

    def test_init_binds_tools(self):
        @tool
        def fake_tool(x: str) -> str:
            """A test tool."""
            return x
        agent = ReActAgent(name="ReAct", model=self.model, tools=[fake_tool])
        self.assertIsNotNone(agent._bound)

    def test_call_agent_handles_llm_error(self):
        self.model.invoke.side_effect = Exception("fail")
        agent = ReActAgent(name="ReAct", model=self.model)
        result = agent._call_agent({"messages": [HumanMessage(content="hi")], "iteration": 0})
        self.assertIn("[LLM error", result["messages"][-1].content)

    def test_execute_tools_unknown_tool(self):
        agent = ReActAgent(name="ReAct", model=self.model)
        msg = AIMessage(content="", tool_calls=[{"name": "no_such_tool", "args": {}, "id": "1", "type": "tool_call"}])
        result = agent._execute_tools({"messages": [msg]})
        self.assertIn("Unknown tool", result["messages"][-1].content)

    def test_route_has_tool_calls(self):
        agent = ReActAgent(name="ReAct", model=self.model)
        msg = AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1", "type": "tool_call"}])
        route = agent._route({"messages": [msg], "iteration": 0})
        self.assertEqual(route, "continue")

    def test_route_no_tool_calls(self):
        agent = ReActAgent(name="ReAct", model=self.model)
        msg = AIMessage(content="final answer")
        route = agent._route({"messages": [msg], "iteration": 0})
        self.assertEqual(route, "end")

    def test_route_over_limit(self):
        agent = ReActAgent(name="ReAct", model=self.model, max_iterations=3)
        msg = AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1", "type": "tool_call"}])
        route = agent._route({"messages": [msg], "iteration": 3})
        self.assertEqual(route, "end")


class TestReflectionAgent(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock()

    def test_generate_draft(self):
        self.model.invoke.return_value = AIMessage(content="draft answer")
        agent = ReflectionAgent(name="Ref", model=self.model)
        state = agent._init_state("what is AI?")
        result = agent._generate_draft(state)
        self.assertEqual(result["draft"], "draft answer")

    def test_generate_draft_handles_error(self):
        self.model.invoke.side_effect = Exception("fail")
        agent = ReflectionAgent(name="Ref", model=self.model)
        result = agent._generate_draft({"messages": [HumanMessage(content="q")], "draft": "", "iteration": 0})
        self.assertEqual(result["draft"], "")

    def test_reflect_empty_draft(self):
        agent = ReflectionAgent(name="Ref", model=self.model)
        state = {"messages": [HumanMessage(content="q")], "draft": "", "iteration": 0}
        result = agent._reflect(state)
        self.assertIn("PASS", result["messages"][-1].content)

    def test_route_pass(self):
        agent = ReflectionAgent(name="Ref", model=self.model)
        msg = AIMessage(content="结论：PASS")
        route = agent._route({"messages": [msg], "iteration": 0})
        self.assertEqual(route, "end")

    def test_route_fail_can_revise(self):
        agent = ReflectionAgent(name="Ref", model=self.model, max_iterations=3)
        msg = AIMessage(content="评分：5/10\n结论：FAIL")
        route = agent._route({"messages": [msg], "iteration": 1})
        self.assertEqual(route, "revise")

    def test_route_fail_exhausted(self):
        agent = ReflectionAgent(name="Ref", model=self.model, max_iterations=3)
        msg = AIMessage(content="结论：FAIL")
        route = agent._route({"messages": [msg], "iteration": 3})
        self.assertEqual(route, "end")


class TestPlanAndSolveAgent(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock()

    def test_plan(self):
        self.model.invoke.return_value = AIMessage(content="PLAN:\n1. step a\n2. step b")
        agent = PlanAndSolveAgent(name="PAS", model=self.model, max_steps=5)
        state = agent._init_state("make a plan")
        result = agent._do_plan(state)
        self.assertEqual(len(result["plan"]), 2)

    def test_plan_handles_error(self):
        self.model.invoke.side_effect = Exception("fail")
        agent = PlanAndSolveAgent(name="PAS", model=self.model)
        result = agent._do_plan({"messages": [HumanMessage(content="q")], "plan": [], "plan_text": "",
                                  "current_step": 0, "step_results": [], "iteration": 0})
        self.assertEqual(result["plan"], [])

    def test_plan_empty_steps_fallback(self):
        self.model.invoke.return_value = AIMessage(content="just a text response")
        agent = PlanAndSolveAgent(name="PAS", model=self.model)
        state = agent._init_state("q")
        result = agent._do_plan(state)
        self.assertGreaterEqual(len(result["plan"]), 1)

    def test_solve(self):
        self.model.invoke.return_value = AIMessage(content="step result")
        agent = PlanAndSolveAgent(name="PAS", model=self.model)
        state = {"messages": [HumanMessage(content="q")], "plan": ["step1"], "plan_text": "PLAN:\n1. step1",
                 "current_step": 0, "step_results": [], "iteration": 0}
        result = agent._do_solve(state)
        self.assertEqual(result["step_results"], ["step result"])
        self.assertEqual(result["current_step"], 1)

    def test_solve_beyond_steps(self):
        agent = PlanAndSolveAgent(name="PAS", model=self.model)
        state = {"messages": [HumanMessage(content="q")], "plan": ["s1"], "plan_text": "",
                 "current_step": 5, "step_results": [], "iteration": 0}
        result = agent._do_solve(state)
        self.assertEqual(result["current_step"], 5)

    def test_after_solve_remaining(self):
        agent = PlanAndSolveAgent(name="PAS", model=self.model)
        route = agent._after_solve({"plan": ["a", "b", "c"], "current_step": 1, "step_results": []})
        self.assertEqual(route, "solve")

    def test_after_solve_done(self):
        agent = PlanAndSolveAgent(name="PAS", model=self.model)
        route = agent._after_solve({"plan": ["a", "b"], "current_step": 2, "step_results": ["r1", "r2"]})
        self.assertEqual(route, "refine")

    def test_refine(self):
        self.model.invoke.return_value = AIMessage(content="refined answer")
        agent = PlanAndSolveAgent(name="PAS", model=self.model)
        state = {"messages": [HumanMessage(content="q")], "step_results": ["step1 result"], "plan": [],
                 "plan_text": "", "current_step": 1, "iteration": 0}
        result = agent._do_refine(state)
        self.assertIn("refined", result["messages"][-1].content)

    def test_refine_empty_results(self):
        agent = PlanAndSolveAgent(name="PAS", model=self.model)
        state = {"messages": [HumanMessage(content="q")], "step_results": [], "plan": [],
                 "plan_text": "", "current_step": 0, "iteration": 0}
        result = agent._do_refine(state)
        self.assertEqual(result["messages"][-1].content, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
