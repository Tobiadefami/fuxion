from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
import os

HUGGING_FACE_API_KEY = os.getenv("HUGGINGFACE_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION")


MODEL_MAP = {
    "gpt-3.5-turbo": ChatOpenAI,
    "gpt-4": ChatOpenAI,
    "gpt-4o": ChatOpenAI,
    "gpt-4o-mini": ChatOpenAI,
}


def get_model(model_name: str, temperature: float, **kwargs) -> OpenAI:
    model_function = MODEL_MAP[model_name]

    if model_function is None:
        raise ValueError(f"Model {model_name} not found in model map, use one of {MODEL_MAP.keys()}")

    return model_function(
        model=model_name,
        temperature=temperature,
        **kwargs,
    )
