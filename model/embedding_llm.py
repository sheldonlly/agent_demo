from langchain.embeddings import init_embeddings
import os
import logging
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()

class EMBEDDING_LLM():
    def __init__(self,
                 api_key: str | None = None,
                 model: str | None = None,
                 base_url: str | None = None,
                 provider: str | None = None):
        self.api_key = api_key or os.getenv("EMBEDDING_LLM_API_KEY")
        if self.api_key is None:
            logger.error("EMBEDDING LLM API key not defined")
        self.model = model or os.getenv("EMBEDDING_LLM_MODEL_NAME")
        if self.model is None:
            logger.error("EMBEDDING LLM model not defined")
        self.base_url = base_url or os.getenv("EMBEDDING_LLM_BASE_URL")
        if self.base_url is None:
            logger.error("EMBEDDING LLM base url not defined")
        self.provider = provider or os.getenv("EMBEDDING_LLM_PROVIDER")
        if self.provider is None:
            logger.error("EMBEDDING LLM provider not defined")

        self.llm = self.create_embedding_llm()
        if self.llm is None:
            logger.error("EMBEDDING LLM model create failed")

    def create_embedding_llm(self):
        self.llm = init_embeddings(
            model = self.model,
            api_key =self.api_key,
            base_url = self.base_url,
            provider = self.provider
        )

        return self.llm

    def embbedding_vector(self, message: str):
        return self.llm.embed_query(message)

    def embbedding_vectors(self, messages: list[str]):
        return self.llm.embed_documents(messages)

if __name__ == "__main__":
    print("test EMBEDDING LLM")
    my_embedding_llm = EMBEDDING_LLM()
    vector = my_embedding_llm.embbedding_vector("你好，请你帮我写一个生日计划的plan")
    print(vector)