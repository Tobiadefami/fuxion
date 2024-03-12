from langchain_openai import OpenAI, ChatOpenAI

MODEL_MAP = {
    "gpt-3.5-turbo": ChatOpenAI,
    "gpt-4": ChatOpenAI,
    "gpt-4-1106-preview": ChatOpenAI,
    "gpt-3.5-turbo-instruct":OpenAI
}


def get_model(model_name: str, temperature: float, cache: bool, **kwargs) -> OpenAI:
    model_function = MODEL_MAP[model_name]
    if model_function is None:
        raise ValueError(f"Model {model_name} not found in model map")
    return model_function(
        model=model_name, temperature=temperature, cache=cache, **kwargs
    )
