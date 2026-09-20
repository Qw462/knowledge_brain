import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RerankerConfig:
    text_reranker_api_key:str
    text_reranker_model:str
    text_reranker_instruct:str

reranker_config=RerankerConfig(
    text_reranker_api_key=os.getenv("OPENAI_API_KEY"),
    text_reranker_model=os.getenv("TEXT_RERANK_MODEL"),
    text_reranker_instruct=os.getenv("TEXT_RERANKER_INSTRUCT")
)
