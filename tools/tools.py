import logging
import numexpr
from langchain_core.tools import tool
from RAG.rag import RAG_Manager

logger = logging.getLogger(__name__)


@tool
def get_weather(city: str) -> str:
    """获取指定城市的当前天气情况"""
    return f"{city}天气晴朗，温度为25摄氏度"


@tool
def caculate(expression: str) -> str:
    """计算数学表达式的值，支持四则运算、幂运算等。传入表达式字符串，如 '2 + 3 * 4'"""
    try:
        result = numexpr.evaluate(expression)
        return str(result)
    except Exception as e:
        logger.warning("caculate failed: %s", e)
        return f"Error: {e}"


@tool
def search_knowledge(query: str) -> str:
    """搜索本地知识库，返回与查询相关的信息片段（RAG 检索）"""
    try:
        rag = RAG_Manager(memory_mode=True)
        results = rag.query(query, top_k=2)
        if not results:
            return "未找到相关信息。"
        parts = []
        for r in results:
            content = r.get("payload", {}).get("content", "")
            score = r.get("score", 0)
            parts.append(f"[相关度 {score:.3f}] {content}")
        return "\n\n".join(parts)
    except Exception as e:
        logger.warning("search_knowledge failed: %s", e)
        return f"Search error: {e}"


@tool
def calculator(expression: str) -> str:
    """计算数学表达式，返回数值结果。支持 + - * / ** 等运算符"""
    return caculate(expression)
