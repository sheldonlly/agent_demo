from langchain.chat_models import init_chat_model
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
import os
import logging
logger = logging.getLogger(__name__)

from pydantic.v1 import create_model

load_dotenv()

class LLM():
    def __init__(self,
                 api_key: str | None = None,
                 model: str | None = None,
                 base_url: str | None = None,
                 provider: str | None = None):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        if self.api_key is None:
            logger.error("LLM API key not defined")
        self.model = model or os.getenv("LLM_MODEL_NAME")
        if self.model is None:
            logger.error("LLM model not defined")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        if self.base_url is None:
            logger.error("LLM base url not defined")
        self.provider = provider or os.getenv("LLM_PROVIDER")
        if self.provider is None:
            logger.error("LLM provider not defined")

        self.llm = self.create_llm()
        if self.llm is None:
            logger.error("LLM model create failed")

    def create_llm(self):
        self.llm = init_chat_model(
            model = self.model,
            api_key =self.api_key,
            base_url = self.base_url,
            model_provider = self.provider
        )

        return self.llm

    def run_llm(self, message: str):
        response = self.llm.invoke(message)
        return response

    def run_llm_stream(self, message: str):
        response = self.llm.stream(message)
        for chunk in response:
            yield chunk

if __name__ == "__main__":
    print("test LLM")
    my_llm = LLM()
    response = my_llm.run_llm_stream("你好，请你帮我写一个生日计划的plan")
    for chunk in response:
        print(chunk.content, end="", flush=True)
