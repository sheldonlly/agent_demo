# agent类创建

from abc import ABC, abstractmethod
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools.structured import BaseTool
class BaseAgent(ABC):
    def __init__(self,
                 name: str,
                 model: str | BaseChatModel,
                 tools: list[str | BaseTool]):
        self.name = name
        self.model = model
        self.tools = tools

    @abstractmethod
    def run(self, query: str):
        pass

class ReActAgent(BaseAgent):
    def __init__(self,
                 name: str,
                 model: str,
                 tools: list[str | BaseTool]):
        super.__init__(name, model, tools)

class ReflectionAgent(BaseAgent):
    def __init__(self,
                 name: str,
                 model: str,
                 tools: list[str | BaseTool]):
        super.__init__(name, model, tools)

class PlanAndSolveAgent(BaseAgent):
    def __init__(self,
                 name: str,
                 model: str,
                 tools: list[str | BaseTool]):
        super.__init__(name, model, tools)