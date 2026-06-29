import logging
import re
from typing import Any, Callable

from context import ContextManager

logger = logging.getLogger(__name__)

HIGH_RISK_KEYWORDS: list[str] = [
    "delete", "drop ", "truncate", "rm -rf", "shutdown",
    "reboot", "format", "dd if=", "chmod 777",
    "exec(", "eval(", "subprocess", "os.system",
    "pickle.load", "shelve.open",
]

PreHook = Callable[[str, list[dict[str, Any]] | None], str | None]
PostHook = Callable[[str, str], str | None]


class Middleware:
    def __init__(self, context_manager: ContextManager) -> None:
        self.context = context_manager
        self._pre_hooks: list[PreHook] = []
        self._post_hooks: list[PostHook] = []
        logger.info("Middleware ready")

    # ── hook registration ─────────────────────────────────────────────

    def register_pre_hook(self, hook: PreHook) -> None:
        self._pre_hooks.append(hook)
        logger.debug("registered pre-hook: %s", getattr(hook, "__name__", "?"))

    def register_post_hook(self, hook: PostHook) -> None:
        self._post_hooks.append(hook)
        logger.debug("registered post-hook: %s", getattr(hook, "__name__", "?"))

    # ── processing pipeline ───────────────────────────────────────────

    def pre_process(
        self,
        query: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> str | None:
        blocked = self._check_high_risk(tool_calls)
        if blocked:
            return blocked

        for hook in self._pre_hooks:
            try:
                result = hook(query, tool_calls)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning("pre-hook %s failed: %s", getattr(hook, "__name__", "?"), e)

        return None

    def post_process(self, query: str, response: str) -> str:
        for hook in self._post_hooks:
            try:
                result = hook(query, response)
                if result is not None:
                    response = result
            except Exception as e:
                logger.warning("post-hook %s failed: %s", getattr(hook, "__name__", "?"), e)
        return response

    # ── high-risk detection ───────────────────────────────────────────

    def _check_high_risk(self, tool_calls: list[dict[str, Any]] | None) -> str | None:
        if not tool_calls:
            return None
        for tc in tool_calls:
            name = tc.get("name", "")
            args = str(tc.get("args", {}))
            combined = f"{name} {args}".lower()

            for kw in HIGH_RISK_KEYWORDS:
                if kw.lower() in combined:
                    logger.warning("High-risk action blocked | keyword=%s action=%s", kw, name)
                    if not self.context.request_approval(tc):
                        return (
                            f"[Blocked] The action '{name}' was identified as high-risk "
                            f"(triggered keyword: '{kw}') and has been blocked. "
                            "Manual approval is required."
                        )
        return None

    # ── built-in hooks ────────────────────────────────────────────────

    @staticmethod
    def sanitize_output_hook(query: str, response: str) -> str | None:
        if response.startswith("[LLM error") or response.startswith("Execution error"):
            logger.warning("response contains error, returning as-is")
        return None

    @staticmethod
    def log_interaction_hook(query: str, response: str) -> str | None:
        logger.info("Interaction | query=%s resp_len=%d", query[:80], len(response))
        return None
