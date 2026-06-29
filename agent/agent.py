import logging
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

# ── make project root importable (so `from prompt import ...` works) ──
_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from prompt import (
    DRAFT_PROMPT,
    REFLECT_PROMPT,
    REVISION_PROMPT,
    SOLVE_PROMPT,
    REFINE_PROMPT,
)

logger = logging.getLogger(__name__)

_REACT_SYSTEM_PROMPT = (
    "你是⼀个智能AI助手，可以使用提供的工具来帮助用户。"
    "如果需要实时信息或执行计算，请使用工具。"
    "每次调用一个工具，等待结果后再决定下一步。"
    "如果已经获得足够信息，请直接回答用户的问题，最终答案用中文回复。"
)


class _BaseState(TypedDict):
    messages: list[BaseMessage]
    iteration: int


class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        model: BaseChatModel,
        tools: list[BaseTool] | None = None,
        max_iterations: int = 10,
    ) -> None:
        self.name = name
        self.model = model
        self.tools = tools or []
        self.max_iterations = max_iterations
        self._graph: CompiledStateGraph | None = None
        logger.info(
            "[%s] init | model=%s | tools=[%s] | max_iterations=%d",
            name,
            model.__class__.__name__,
            ", ".join(t.name for t in self.tools),
            max_iterations,
        )

    @abstractmethod
    def build_graph(self) -> CompiledStateGraph:
        ...

    def run(self, query: str) -> str:
        logger.info("[%s] run | query=%s", self.name, query[:120])
        if self._graph is None:
            logger.debug("[%s] build_graph ...", self.name)
            self._graph = self.build_graph()
        try:
            result = self._graph.invoke(self._init_state(query))
            answer = self._extract_answer(result)
            logger.info("[%s] done | answer_len=%d", self.name, len(answer))
            return answer
        except Exception:
            logger.exception("[%s] run failed", self.name)
            return "Execution error."

    def _init_state(self, query: str) -> dict:
        return {"messages": [HumanMessage(content=query)], "iteration": 0}

    def _extract_answer(self, state: dict) -> str:
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return str(msg.content)
        if messages:
            last = messages[-1]
            content = getattr(last, "content", str(last))
            logger.warning("[%s] fallback: last msg is %s", self.name, type(last).__name__)
            return str(content) if content else "Empty response"
        logger.warning("[%s] no messages in state", self.name)
        return "No response."


# ═══════════════════════════════════════════════════════════════════════════════
# ReAct Agent  (tool-calling 版本)
# ═══════════════════════════════════════════════════════════════════════════════

class ReActState(_BaseState):
    pass


class ReActAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        model: BaseChatModel,
        tools: list[BaseTool] | None = None,
        max_iterations: int = 10,
    ) -> None:
        super().__init__(name, model, tools, max_iterations)
        self._bound = self.model.bind_tools(self.tools) if self.tools else self.model
        logger.info("[%s] bind_tools=%d", name, len(self.tools))

    def build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(ReActState)
        builder.add_node("agent", self._call_agent)
        builder.add_node("tools", self._execute_tools)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent", self._route, {"continue": "tools", "end": END},
        )
        builder.add_edge("tools", "agent")
        return builder.compile()

    def _call_agent(self, state: ReActState) -> dict:
        iteration = state.get("iteration", 0)
        try:
            response = self._bound.invoke(
                [SystemMessage(content=_REACT_SYSTEM_PROMPT)] + state["messages"]
            )
        except Exception as e:
            logger.exception("[%s] LLM call failed at iter %d", self.name, iteration)
            return {
                "messages": state["messages"]
                + [AIMessage(content=f"[LLM error: {e}]")],
                "iteration": iteration + 1,
            }

        if not isinstance(response, AIMessage):
            logger.warning(
                "[%s] unexpected response type=%s at iter %d",
                self.name, type(response).__name__, iteration,
            )
            return {
                "messages": state["messages"]
                + [AIMessage(content=str(response))],
                "iteration": iteration + 1,
            }

        tool_calls = response.tool_calls or []
        if tool_calls:
            names = ", ".join(tc["name"] for tc in tool_calls)
            logger.info(
                "[%s] iter=%d | tool_calls=[%s]",
                self.name, iteration, names,
            )
        else:
            content_preview = (response.content or "")[:100]
            logger.info(
                "[%s] iter=%d | final | content=%s",
                self.name, iteration, content_preview,
            )

        return {
            "messages": state["messages"] + [response],
            "iteration": iteration + 1,
        }

    def _execute_tools(self, state: ReActState) -> dict:
        last = state["messages"][-1]
        messages = state["messages"]

        for tc in last.tool_calls:
            tool = next((t for t in self.tools if t.name == tc["name"]), None)
            if tool is None:
                logger.warning("[%s] unknown tool '%s'", self.name, tc["name"])
                messages += [ToolMessage(content=f"Unknown tool: {tc['name']}", tool_call_id=tc["id"])]
                continue

            args_preview = str(tc.get("args", {}))[:200]
            logger.info(
                "[%s] tool=%s | args=%s",
                self.name, tc["name"], args_preview,
            )
            try:
                result = tool.invoke(tc["args"])
                logger.debug("[%s] tool=%s | result=%s", self.name, tc["name"], str(result)[:300])
                messages += [ToolMessage(content=str(result), tool_call_id=tc["id"])]
            except Exception as e:
                logger.exception("[%s] tool=%s failed", self.name, tc["name"])
                messages += [ToolMessage(content=f"Error: {e}", tool_call_id=tc["id"])]

        return {"messages": messages}

    def _route(self, state: ReActState) -> Literal["continue", "end"]:
        last = state["messages"][-1] if state["messages"] else None
        has_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
        over_limit = state.get("iteration", 0) >= self.max_iterations
        if has_calls and not over_limit:
            logger.debug("[%s] route → continue (tools pending)", self.name)
            return "continue"
        if over_limit:
            logger.warning("[%s] route → end (max_iterations=%d reached)", self.name, self.max_iterations)
        else:
            logger.debug("[%s] route → end (final answer)", self.name)
        return "end"


# ═══════════════════════════════════════════════════════════════════════════════
# Reflection Agent
# ═══════════════════════════════════════════════════════════════════════════════

class ReflectionState(_BaseState):
    draft: str


class ReflectionAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        model: BaseChatModel,
        tools: list[BaseTool] | None = None,
        max_iterations: int = 3,
    ) -> None:
        super().__init__(name, model, tools, max_iterations)

    def build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(ReflectionState)
        builder.add_node("draft", self._generate_draft)
        builder.add_node("reflect", self._reflect)
        builder.add_node("revise", self._revise)
        builder.add_edge(START, "draft")
        builder.add_edge("draft", "reflect")
        builder.add_conditional_edges("reflect", self._route, {"revise": "revise", "end": END})
        builder.add_edge("revise", "reflect")
        return builder.compile()

    def _init_state(self, query: str) -> dict:
        return {"messages": [HumanMessage(content=query)], "draft": "", "iteration": 0}

    def _extract_answer(self, state: dict) -> str:
        draft = state.get("draft", "")
        if draft:
            return draft
        return super()._extract_answer(state)

    def _generate_draft(self, state: ReflectionState) -> dict:
        try:
            prompt = DRAFT_PROMPT.format(question=state["messages"][0].content)
            resp = self.model.invoke([HumanMessage(content=prompt)])
        except Exception as e:
            logger.exception("[%s] draft failed", self.name)
            return {
                "messages": state["messages"] + [AIMessage(content=f"[Draft error: {e}]")],
                "draft": "",
                "iteration": state["iteration"] + 1,
            }

        text = str(resp.content) if resp and resp.content else ""
        logger.info("[%s] draft | len=%d | preview=%s", self.name, len(text), text[:120])
        return {
            "messages": state["messages"] + [resp],
            "draft": text,
            "iteration": state["iteration"] + 1,
        }

    def _reflect(self, state: ReflectionState) -> dict:
        if not state["draft"]:
            logger.warning("[%s] reflect skipped — empty draft", self.name)
            return {"messages": state["messages"] + [AIMessage(content="结论：PASS\n理由：草稿为空，无需评审。")]}

        try:
            prompt = REFLECT_PROMPT.format(
                question=state["messages"][0].content,
                answer=state["draft"],
            )
            resp = self.model.invoke([HumanMessage(content=prompt)])
        except Exception as e:
            logger.exception("[%s] reflect failed", self.name)
            resp = AIMessage(content=f"[Reflect error: {e}]")

        text = str(resp.content) if resp and resp.content else ""
        score = _extract_score(text)
        passed = "PASS" in text.upper()
        logger.info("[%s] reflect | score=%s | PASS=%s | preview=%s", self.name, score, passed, text[:150])
        return {"messages": state["messages"] + [resp]}

    def _revise(self, state: ReflectionState) -> dict:
        if not state["draft"]:
            logger.warning("[%s] revise skipped — empty draft", self.name)
            return {
                "messages": state["messages"] + [AIMessage(content="")],
                "draft": "",
                "iteration": state["iteration"] + 1,
            }

        reflection_text = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                reflection_text = str(msg.content)
                break

        try:
            prompt = REVISION_PROMPT.format(
                question=state["messages"][0].content,
                answer=state["draft"],
                reflection=reflection_text,
            )
            resp = self.model.invoke([HumanMessage(content=prompt)])
        except Exception as e:
            logger.exception("[%s] revise failed", self.name)
            return {
                "messages": state["messages"] + [AIMessage(content=f"[Revise error: {e}]")],
                "draft": state["draft"],
                "iteration": state["iteration"] + 1,
            }

        text = str(resp.content) if resp and resp.content else ""
        logger.info("[%s] revise | len=%d | preview=%s", self.name, len(text), text[:120])
        return {
            "messages": state["messages"] + [resp],
            "draft": text,
            "iteration": state["iteration"] + 1,
        }

    def _route(self, state: ReflectionState) -> Literal["revise", "end"]:
        last = state["messages"][-1] if state["messages"] else None
        content = str(last.content) if last and last.content else ""
        failed = "FAIL" in content.upper()
        can_revise = state.get("iteration", 0) < self.max_iterations

        if failed and can_revise:
            logger.debug("[%s] route → revise (FAIL, iter=%d)", self.name, state.get("iteration", 0))
            return "revise"
        if failed:
            logger.warning("[%s] route → end (FAIL but iter exhausted)", self.name)
        else:
            logger.debug("[%s] route → end (PASS)", self.name)
        return "end"


def _extract_score(text: str) -> str:
    m = re.search(r"(?:评分|分数|得分)[：:]\s*(\d+\.?\d*)/?\d*", text)
    return m.group(1) if m else "N/A"


# ═══════════════════════════════════════════════════════════════════════════════
# Plan-and-Solve Agent
# ═══════════════════════════════════════════════════════════════════════════════

class PlanAndSolveState(_BaseState):
    plan: list[str]
    plan_text: str
    current_step: int
    step_results: list[str]


class PlanAndSolveAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        model: BaseChatModel,
        tools: list[BaseTool] | None = None,
        max_iterations: int = 10,
        max_steps: int = 5,
    ) -> None:
        super().__init__(name, model, tools, max_iterations)
        self.max_steps = max_steps
        self._plan_prompt = (
            "你是一名专业的任务规划专家（Planner）。\n"
            "你的任务是：将复杂问题拆解为清晰、可执行的步骤计划。\n"
            "要求：\n"
            "1. 将问题拆解为 3~{max_steps} 个步骤\n"
            "2. 每一步必须是可执行的\n"
            "3. 不要解决问题，只输出计划\n"
            "4. 步骤要有逻辑顺序\n"
            "5. 避免冗余步骤\n"
            "输出格式：\n"
            "PLAN:\n"
            "1. ...\n"
            "2. ...\n"
            "3. ...\n"
            "用户问题：\n{question}"
        )

    def build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(PlanAndSolveState)
        builder.add_node("plan", self._do_plan)
        builder.add_node("solve", self._do_solve)
        builder.add_node("refine", self._do_refine)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "solve")
        builder.add_conditional_edges("solve", self._after_solve, {"solve": "solve", "refine": "refine"})
        builder.add_edge("refine", END)
        return builder.compile()

    def _init_state(self, query: str) -> dict:
        return {
            "messages": [HumanMessage(content=query)],
            "plan": [],
            "plan_text": "",
            "current_step": 0,
            "step_results": [],
            "iteration": 0,
        }

    # ── plan ─────────────────────────────────────────────────────────────

    def _do_plan(self, state: PlanAndSolveState) -> dict:
        try:
            prompt = self._plan_prompt.format(
                question=state["messages"][0].content,
                max_steps=self.max_steps,
            )
            resp = self.model.invoke([HumanMessage(content=prompt)])
        except Exception as e:
            logger.exception("[%s] plan failed", self.name)
            return {
                "messages": state["messages"] + [AIMessage(content=f"[Plan error: {e}]")],
                "plan": [],
                "plan_text": "",
                "iteration": state["iteration"] + 1,
            }

        plan_text = str(resp.content) if resp and resp.content else ""
        steps = _parse_steps(plan_text)

        if len(steps) > self.max_steps:
            logger.warning(
                "[%s] plan has %d steps, truncating to %d",
                self.name, len(steps), self.max_steps,
            )
            steps = steps[:self.max_steps]

        if not steps:
            logger.warning("[%s] plan produced 0 steps, using raw text as 1 step", self.name)
            steps = [plan_text[:300]] if plan_text.strip() else ["Answer the question."]

        logger.info("[%s] plan | steps=%d %s", self.name, len(steps), [s[:60] for s in steps])
        return {
            "messages": state["messages"] + [resp],
            "plan": steps,
            "plan_text": plan_text,
            "iteration": state["iteration"] + 1,
        }

    # ── solve ────────────────────────────────────────────────────────────

    def _do_solve(self, state: PlanAndSolveState) -> dict:
        step_idx = state["current_step"]
        steps = state["plan"]
        results = state["step_results"]

        if step_idx >= len(steps):
            logger.warning("[%s] solve called but all steps done", self.name)
            return {
                "messages": state["messages"] + [AIMessage(content="")],
                "step_results": results,
                "current_step": step_idx,
                "iteration": state["iteration"] + 1,
            }

        current_step_text = steps[step_idx]
        context = _build_step_context(steps, results)

        exec_question = (
            f"{state['messages'][0].content}\n\n"
            f"Current step ({step_idx + 1}/{len(steps)}): {current_step_text}"
        )
        if context:
            exec_question += f"\n\nPrevious results:\n{context}"

        try:
            prompt = SOLVE_PROMPT.format(
                plan=state["plan_text"],
                question=exec_question,
            )
            resp = self.model.invoke([HumanMessage(content=prompt)])
        except Exception as e:
            logger.exception("[%s] solve step %d failed", self.name, step_idx)
            resp = AIMessage(content=f"[Step {step_idx + 1} error: {e}]")

        text = str(resp.content) if resp and resp.content else ""
        logger.info(
            "[%s] solve | step %d/%d | len=%d | preview=%s",
            self.name, step_idx + 1, len(steps), len(text), text[:120],
        )

        return {
            "messages": state["messages"] + [resp],
            "step_results": results + [text],
            "current_step": step_idx + 1,
            "iteration": state["iteration"] + 1,
        }

    # ── refine ───────────────────────────────────────────────────────────

    def _do_refine(self, state: PlanAndSolveState) -> dict:
        if not state["step_results"]:
            logger.warning("[%s] refine skipped — no step results", self.name)
            return {"messages": state["messages"] + [AIMessage(content="")]}

        full = "\n\n".join(state["step_results"])
        try:
            prompt = REFINE_PROMPT.format(answer=full)
            resp = self.model.invoke([HumanMessage(content=prompt)])
        except Exception as e:
            logger.exception("[%s] refine failed", self.name)
            resp = AIMessage(content=full)

        text = str(resp.content) if resp and resp.content else full
        logger.info("[%s] refine | len=%d | preview=%s", self.name, len(text), text[:120])
        return {"messages": state["messages"] + [resp]}

    # ── router ───────────────────────────────────────────────────────────

    def _after_solve(self, state: PlanAndSolveState) -> Literal["solve", "refine"]:
        remaining = len(state["plan"]) - state["current_step"]
        if remaining > 0:
            logger.debug("[%s] %d step(s) remaining → solve", self.name, remaining)
            return "solve"
        logger.debug("[%s] all steps done → refine", self.name)
        return "refine"


# ─── Shared helpers ──────────────────────────────────────────────────────────

def _parse_steps(text: str) -> list[str]:
    lines = text.strip().split("\n")
    steps: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = re.match(
            r"^(?:\d+[.．:、)\-]\s*|[-*]\s+|[Ss]tep\s*\d+[.．:、)\-]\s*)(.*)",
            line,
        )
        if m:
            content = m.group(1).strip()
            if content:
                steps.append(content)
    return steps


def _build_step_context(steps: list[str], results: list[str]) -> str:
    if not results:
        return ""
    lines = [
        f"Step {i + 1}: {s}\nResult: {r}"
        for i, (s, r) in enumerate(zip(steps[:len(results)], results))
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# __main__  —— 集成测试
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from dotenv import load_dotenv
    from model.chat_llm import LLM
    from tools.tools import get_weather, caculate

    load_dotenv()

    log_format = "%(asctime)s | %(name)s | %(levelname)-5s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, force=True)

    try:
        llm = LLM().llm
        logger.info("LLM ready | model=%s", getattr(llm, "model_name", "?"))
    except Exception as e:
        logger.exception("LLM init failed")
        print(f"FATAL: cannot init LLM — {e}")
        raise SystemExit(1)

    test_cases = [
        ("ReActAgent",
         lambda: ReActAgent(name="ReActDemo", model=llm, tools=[get_weather, caculate]),
         "北京的天气怎么样？"),
        ("ReflectionAgent",
         lambda: ReflectionAgent(name="ReflectionDemo", model=llm),
         "请解释什么是机器学习"),
        ("PlanAndSolveAgent",
         lambda: PlanAndSolveAgent(name="PlanAndSolveDemo", model=llm),
         "请为我制定一个为期三天的北京旅游计划"),
    ]

    for label, factory, query in test_cases:
        print(f"\n{'=' * 60}")
        print(f"  Test: {label}")
        print(f"{'=' * 60}")
        try:
            agent = factory()
            result = agent.run(query)
            preview = result[:600]
            print(f"\nAnswer:\n{preview}")
            if len(result) > 600:
                print("...(truncated)")
        except Exception as e:
            print(f"\n[FAIL] {label}: {e}")
            logger.exception("[%s] test failed", label)
