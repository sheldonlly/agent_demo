import logging
import re
from abc import ABC, abstractmethod
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

from prompt import (
    REACT_PROMPT,
    DRAFT_PROMPT,
    REFLECT_PROMPT,
    REVISION_PROMPT,
    PLAN_PROMPT,
    SOLVE_PROMPT,
    REFINE_PROMPT,
)

logger = logging.getLogger(__name__)


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

    @abstractmethod
    def build_graph(self) -> CompiledStateGraph:
        ...

    def run(self, query: str) -> str:
        if self._graph is None:
            self._graph = self.build_graph()
        try:
            result = self._graph.invoke(self._init_state(query))
            return self._extract_answer(result)
        except Exception:
            logger.exception(f"{self.name} execution failed")
            return "An error occurred during execution."

    def _init_state(self, query: str) -> dict:
        return {"messages": [HumanMessage(content=query)], "iteration": 0}

    def _extract_answer(self, state: dict) -> str:
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return str(msg.content)
        return str(messages[-1]) if messages else "No response."


class ReActState(_BaseState):
    pass


class ReActAgent(BaseAgent):
    def build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(ReActState)
        builder.add_node("agent", self._call_agent)
        builder.add_node("tools", self._execute_tools)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent", self._route, {"continue": "tools", "end": END}
        )
        builder.add_edge("tools", "agent")
        return builder.compile()

    def _call_agent(self, state: ReActState) -> dict:
        query = state["messages"][0].content if state["messages"] else ""

        tool_descs = "\n".join(
            f"{t.name}: {t.description}" for t in self.tools
        )
        tool_names = ", ".join(t.name for t in self.tools)
        system_prompt = REACT_PROMPT.format(
            tools=tool_descs, tool_names=tool_names, input=query
        )

        response = self.model.invoke(
            [SystemMessage(content=system_prompt)] + state["messages"]
        )

        return {
            "messages": state["messages"] + [response],
            "iteration": state["iteration"] + 1,
        }

    def _execute_tools(self, state: ReActState) -> dict:
        last_message = state["messages"][-1]
        messages = state["messages"]

        for tool_call in last_message.tool_calls:
            tool = next(
                (t for t in self.tools if t.name == tool_call["name"]), None
            )
            if tool:
                try:
                    result = tool.invoke(tool_call["args"])
                    messages = messages + [
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"],
                        )
                    ]
                except Exception as e:
                    messages = messages + [
                        ToolMessage(
                            content=f"Tool error: {e}",
                            tool_call_id=tool_call["id"],
                        )
                    ]
            else:
                messages = messages + [
                    ToolMessage(
                        content=f"Unknown tool: '{tool_call['name']}'",
                        tool_call_id=tool_call["id"],
                    )
                ]

        return {"messages": messages}

    def _route(self, state: ReActState) -> Literal["continue", "end"]:
        last = state["messages"][-1] if state["messages"] else None
        has_tool_calls = isinstance(last, AIMessage) and last.tool_calls
        return "continue" if (has_tool_calls and state["iteration"] < self.max_iterations) else "end"


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
        builder.add_conditional_edges(
            "reflect", self._route, {"revise": "revise", "end": END}
        )
        builder.add_edge("revise", "reflect")
        return builder.compile()

    def _init_state(self, query: str) -> dict:
        return {
            "messages": [HumanMessage(content=query)],
            "draft": "",
            "iteration": 0,
        }

    def _extract_answer(self, state: dict) -> str:
        return state.get("draft", "") or super()._extract_answer(state)

    def _generate_draft(self, state: ReflectionState) -> dict:
        query = state["messages"][0].content
        prompt = DRAFT_PROMPT.format(question=query)
        response = self.model.invoke([HumanMessage(content=prompt)])
        return {
            "messages": state["messages"] + [response],
            "draft": str(response.content),
            "iteration": state["iteration"] + 1,
        }

    def _reflect(self, state: ReflectionState) -> dict:
        query = state["messages"][0].content
        prompt = REFLECT_PROMPT.format(question=query, answer=state["draft"])
        response = self.model.invoke([HumanMessage(content=prompt)])
        return {"messages": state["messages"] + [response]}

    def _revise(self, state: ReflectionState) -> dict:
        query = state["messages"][0].content
        reflection_text = ""
        if state["messages"] and isinstance(state["messages"][-1], AIMessage):
            reflection_text = str(state["messages"][-1].content)
        prompt = REVISION_PROMPT.format(
            question=query, answer=state["draft"], reflection=reflection_text
        )
        response = self.model.invoke([HumanMessage(content=prompt)])
        return {
            "messages": state["messages"] + [response],
            "draft": str(response.content),
            "iteration": state["iteration"] + 1,
        }

    def _route(self, state: ReflectionState) -> Literal["revise", "end"]:
        content = str(state["messages"][-1].content) if state["messages"] else ""
        failed = "FAIL" in content
        can_revise = state["iteration"] < self.max_iterations
        return "revise" if (failed and can_revise) else "end"


class PlanAndSolveState(_BaseState):
    plan: list[str]
    plan_text: str
    current_step: int
    step_results: list[str]


class PlanAndSolveAgent(BaseAgent):
    def build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(PlanAndSolveState)
        builder.add_node("plan", self._do_plan)
        builder.add_node("solve", self._do_solve)
        builder.add_node("refine", self._do_refine)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "solve")
        builder.add_conditional_edges(
            "solve", self._after_solve, {"solve": "solve", "refine": "refine"}
        )
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

    @staticmethod
    def _parse_steps(plan_text: str) -> list[str]:
        lines = plan_text.strip().split("\n")
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

    def _do_plan(self, state: PlanAndSolveState) -> dict:
        query = state["messages"][0].content
        prompt = PLAN_PROMPT.format(question=query)
        response = self.model.invoke([HumanMessage(content=prompt)])
        plan_text = str(response.content)
        return {
            "messages": state["messages"] + [response],
            "plan": self._parse_steps(plan_text),
            "plan_text": plan_text,
            "iteration": state["iteration"] + 1,
        }

    def _do_solve(self, state: PlanAndSolveState) -> dict:
        step_idx = state["current_step"]
        steps = state["plan"]
        results = state["step_results"]

        context_lines = [
            f"Step {i+1}: {s}\nResult: {r}"
            for i, (s, r) in enumerate(zip(steps[:step_idx], results))
        ]
        context = "\n".join(context_lines)
        current = steps[step_idx]
        query = state["messages"][0].content

        exec_question = (
            f"{query}\n\nCurrent step ({step_idx + 1}/{len(steps)}): {current}"
        )
        if context:
            exec_question += f"\n\nPrevious results:\n{context}"

        prompt = SOLVE_PROMPT.format(
            plan=state["plan_text"], question=exec_question
        )
        response = self.model.invoke([HumanMessage(content=prompt)])

        return {
            "messages": state["messages"] + [response],
            "step_results": results + [str(response.content)],
            "current_step": step_idx + 1,
            "iteration": state["iteration"] + 1,
        }

    def _do_refine(self, state: PlanAndSolveState) -> dict:
        full_answer = "\n\n".join(state["step_results"])
        prompt = REFINE_PROMPT.format(answer=full_answer)
        response = self.model.invoke([HumanMessage(content=prompt)])
        return {"messages": state["messages"] + [response]}

    def _after_solve(self, state: PlanAndSolveState) -> Literal["solve", "refine"]:
        return "solve" if state["current_step"] < len(state["plan"]) else "refine"


if __name__ == "__main__":
    from dotenv import load_dotenv
    from model.chat_llm import LLM
    from tools.tools import get_weather, caculate

    load_dotenv()
    llm = LLM().llm

    print("=" * 60)
    print("Test: ReActAgent")
    print("=" * 60)
    react = ReActAgent(name="ReActDemo", model=llm, tools=[get_weather, caculate])
    result = react.run("北京的天气怎么样？")
    print(f"Answer: {result}")
    print()

    print("=" * 60)
    print("Test: ReflectionAgent")
    print("=" * 60)
    reflect = ReflectionAgent(name="ReflectionDemo", model=llm)
    result = reflect.run("请解释什么是机器学习")
    print(f"Answer: {result}")
    print()

    print("=" * 60)
    print("Test: PlanAndSolveAgent")
    print("=" * 60)
    pas = PlanAndSolveAgent(name="PlanAndSolveDemo", model=llm)
    result = pas.run("请为我制定一个为期三天的北京旅游计划")
    print(f"Answer: {result}")
