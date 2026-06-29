import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_proj_root = str(Path(__file__).resolve().parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import log.logconfig  # noqa: F401
from agent.agent import ReActAgent, ReflectionAgent, PlanAndSolveAgent
from context import ContextManager
from memory.memoryManager import MemoryManager
from middleware.middleware import Middleware
from model.chat_llm import LLM as ChatLLM
from RAG.rag import RAG_Manager
from tools.tools import get_weather, caculate, search_knowledge

logger = logging.getLogger(__name__)


class AgentCLI:
    def __init__(self) -> None:
        load_dotenv()
        self._rag: RAG_Manager | None = None
        self._memory = MemoryManager()
        self._context = ContextManager(memory=self._memory)
        self._middleware = Middleware(self._context)
        self._llm = self._init_llm()
        self._agent = self._create_default_agent()

    def _init_llm(self) -> ChatLLM:
        try:
            llm = ChatLLM()
            logger.info("LLM initialized: %s", getattr(llm.llm, "model_name", "unknown"))
            return llm
        except Exception as e:
            logger.error("Failed to initialize LLM: %s", e)
            raise SystemExit(1)

    def _create_default_agent(self, mode: str = "react") -> ReActAgent | ReflectionAgent | PlanAndSolveAgent:
        tools = [get_weather, caculate, search_knowledge]
        kwargs = {
            "name": f"{mode.capitalize()}Agent",
            "model": self._llm.llm,
            "memory": self._memory,
            "rag": self._rag,
            "context": self._context,
            "middleware": self._middleware,
        }

        match mode:
            case "react":
                return ReActAgent(tools=tools, **kwargs)
            case "reflection":
                return ReflectionAgent(**kwargs)
            case "plan":
                return PlanAndSolveAgent(tools=tools, **kwargs)
            case _:
                logger.warning("Unknown mode '%s', falling back to react", mode)
                return ReActAgent(tools=tools, **kwargs)

    def _seed_rag(self, paths: list[str]) -> int:
        texts = []
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    texts.append(f.read())
            except Exception as e:
                logger.error("Failed to read %s: %s", p, e)
        if texts:
            if self._rag is None:
                self._rag = RAG_Manager(memory_mode=True)
                self._context.rag = self._rag
            self._rag.add_documents(texts)
        return len(texts)

    def run_interactive(self, mode: str = "react") -> None:
        self._agent = self._create_default_agent(mode)
        session_id = self._context.session_id

        print(f"\n{'=' * 60}")
        print(f"  AI Agent CLI (mode: {mode})")
        print(f"  Session: {session_id}")
        print(f"  Commands: /help, /mode, /session, /memory, /clear, /export, /quit")
        print(f"{'=' * 60}\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                self._handle_command(user_input)
                continue

            try:
                print("\nAgent: ", end="", flush=True)
                response = self._agent.run(user_input)
                print(response)
                print()
            except Exception as e:
                logger.exception("Run failed")
                print(f"\n[Error] {e}\n")

    def run_once(self, query: str, mode: str = "react") -> str:
        self._agent = self._create_default_agent(mode)
        return self._agent.run(query)

    def _handle_command(self, cmd: str) -> None:
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        match command:
            case "/help":
                print("""
  /help              - Show this help
  /mode <react|reflection|plan> - Switch agent mode
  /session           - Show current session info
  /sessions          - List all sessions
  /switch <id>       - Switch to another session
  /memory            - Show memory stats
  /clear             - Clear conversation history
  /export            - Export conversation as markdown
  /quit              - Exit
                """.strip())

            case "/mode":
                modes = {"react", "reflection", "plan"}
                if arg in modes:
                    self._agent = self._create_default_agent(arg)
                    print(f"Switched to {arg} mode")
                else:
                    print(f"Available modes: {', '.join(sorted(modes))}")

            case "/session":
                info = self._context.get_session_info()
                for k, v in info.items():
                    print(f"  {k}: {v}")

            case "/sessions":
                sessions = self._context.list_sessions()
                if not sessions:
                    print("  No saved sessions")
                else:
                    for s in sessions:
                        print(f"  {s['session_id']} | turns={s.get('turn_count', 0)} | {s.get('created_at', '?')}")

            case "/switch":
                if not arg:
                    print("  Usage: /switch <session_id>")
                elif self._context.switch_session(arg):
                    mode_name = self._agent.__class__.__name__.replace("Agent", "").lower()
                    self._agent = self._create_default_agent(mode_name)
                    print(f"Switched to session {arg}")
                else:
                    print(f"Session {arg} not found")

            case "/memory":
                stats = self._memory.stats()
                for k, v in stats.items():
                    print(f"  {k}: {v}")

            case "/clear":
                self._context.clear_history()
                print("Conversation history cleared")

            case "/export":
                md = self._context.export_history(format="markdown")
                export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
                os.makedirs(export_dir, exist_ok=True)
                export_path = os.path.join(export_dir, f"session_{self._context.session_id}.md")
                with open(export_path, "w", encoding="utf-8") as f:
                    f.write(md)
                print(f"Exported to {export_path}")

            case "/quit":
                print("Goodbye!")
                raise SystemExit(0)

            case _:
                print(f"Unknown command: {command}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Agent CLI - Multi-mode intelligent assistant",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Query to run (omit for interactive mode)",
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["react", "reflection", "plan"],
        default="react",
        help="Agent mode (default: react)",
    )
    parser.add_argument(
        "--seed-rag",
        nargs="+",
        metavar="FILE",
        help="Seed RAG with document files (txt, md)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="AI Agent v0.2.0",
    )

    args = parser.parse_args()

    try:
        cli = AgentCLI()
    except SystemExit:
        return

    if args.seed_rag:
        count = cli._seed_rag(args.seed_rag)
        print(f"Seeded {count} documents into RAG")

    if args.query:
        result = cli.run_once(args.query, mode=args.mode)
        print(result)
    else:
        cli.run_interactive(mode=args.mode)


if __name__ == "__main__":
    main()
