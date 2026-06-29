import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import log.logconfig  # noqa: F401
from tools.tools import get_weather, caculate, calculator, search_knowledge
from langchain_core.tools import BaseTool


class TestToolBasic(unittest.TestCase):
    def test_get_weather_is_tool(self):
        self.assertIsInstance(get_weather, BaseTool)

    def test_get_weather_name(self):
        self.assertEqual(get_weather.name, "get_weather")

    def test_get_weather_result(self):
        result = get_weather.invoke({"city": "Beijing"})
        self.assertIn("Beijing", result)
        self.assertIn("晴朗", result)

    def test_get_weather_different_city(self):
        result = get_weather.invoke({"city": "Shanghai"})
        self.assertIn("Shanghai", result)

    def test_get_weather_empty_city(self):
        result = get_weather.invoke({"city": ""})
        self.assertIn("晴朗", result)

    def test_caculate_is_tool(self):
        self.assertIsInstance(caculate, BaseTool)

    def test_caculate_name(self):
        self.assertEqual(caculate.name, "caculate")

    def test_caculate_simple_addition(self):
        result = caculate.invoke({"expression": "2 + 3"})
        self.assertEqual(float(result), 5.0)

    def test_caculate_complex(self):
        result = caculate.invoke({"expression": "2 * 3 + 4 / 2"})
        self.assertEqual(float(result), 8.0)

    def test_caculate_power(self):
        result = caculate.invoke({"expression": "2 ** 10"})
        self.assertEqual(float(result), 1024.0)

    def test_caculate_division_by_zero(self):
        result = caculate.invoke({"expression": "1 / 0"})
        self.assertIn("Error", result)

    def test_caculate_invalid_expression(self):
        result = caculate.invoke({"expression": "hello + world"})
        self.assertIn("Error", result)

    def test_caculate_float_result(self):
        result = caculate.invoke({"expression": "10 / 3"})
        self.assertAlmostEqual(float(result), 3.3333, places=3)

    def test_caculate_negative_numbers(self):
        result = caculate.invoke({"expression": "-5 + 3"})
        self.assertEqual(float(result), -2.0)

    def test_calculate_is_alias(self):
        self.assertIsInstance(calculator, BaseTool)
        self.assertEqual(calculator.name, "calculator")


class TestSearchKnowledge(unittest.TestCase):
    def test_search_knowledge_is_tool(self):
        self.assertIsInstance(search_knowledge, BaseTool)

    @patch("tools.tools.RAG_Manager")
    def test_search_knowledge_success(self, mock_rag):
        mock_instance = mock_rag.return_value
        mock_instance.query.return_value = [
            {"payload": {"content": "Python is a programming language"}, "score": 0.95}
        ]
        result = search_knowledge.invoke({"query": "Python"})
        self.assertIn("Python", result)
        self.assertIn("0.950", result)

    @patch("tools.tools.RAG_Manager")
    def test_search_knowledge_no_results(self, mock_rag):
        mock_instance = mock_rag.return_value
        mock_instance.query.return_value = []
        result = search_knowledge.invoke({"query": "unknown"})
        self.assertIn("未找到", result)

    @patch("tools.tools.RAG_Manager")
    def test_search_knowledge_error(self, mock_rag):
        mock_rag.side_effect = RuntimeError("RAG unavailable")
        result = search_knowledge.invoke({"query": "test"})
        self.assertIn("Search error", result)

    @patch("tools.tools.RAG_Manager")
    def test_search_knowledge_multiple_results(self, mock_rag):
        mock_instance = mock_rag.return_value
        mock_instance.query.return_value = [
            {"payload": {"content": "Result 1"}, "score": 0.9},
            {"payload": {"content": "Result 2"}, "score": 0.8},
        ]
        result = search_knowledge.invoke({"query": "test"})
        self.assertIn("Result 1", result)
        self.assertIn("Result 2", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
