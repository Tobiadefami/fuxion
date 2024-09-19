from typing import List, Optional, Dict, Any, Type
from pydantic import BaseModel, Field, create_model
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_community.cache import SQLiteCache
import typer
from rich import print
import langchain
from fuxion.dynamic_models import create_dynamic_model
from fuxion.models import get_model





class GeneratorChain:
    def __init__(
        self,
        template_file: str,
        output_structure: Dict[str, Any],
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        verbose: bool = False,
        cache: bool = False
    ):
        self.cache = cache
        if self.cache:
            langchain.llm_cache = SQLiteCache("llm_cache.db")
        self.template = PromptTemplate.from_file(template_file, input_variables=["few_shot"])
        self.model = get_model(model_name=model_name, temperature=temperature)
        self.GeneratedItem = create_dynamic_model(output_structure)
        self.GeneratedOutput = create_model("GeneratedOutput",
            items=(List[self.GeneratedItem], Field(description="List of generated items")))
        self.structured_llm = self.model.with_structured_output(self.GeneratedOutput)
        self.verbose = verbose


    def generate(self, few_shot: str) -> Any:
        prompt = self.template.format(few_shot=few_shot)
        messages = [HumanMessage(content=prompt)]

        # if self.verbose:
        #     print(f"Prompt: {prompt}")

        result = self.structured_llm.invoke(messages)

        return result

def main(
    template_file: str,
    few_shot_example: str,
    output_structure: str,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    verbose: bool = False,
    cache: bool = False
):
    structured_dict = eval(output_structure)
    chain = GeneratorChain(
        template_file=template_file,
        output_structure=structured_dict,
        model_name=model_name,
        temperature=temperature,
        verbose=verbose
    )
    output = chain.generate(few_shot_example)
    print(output)

if __name__ == "__main__":
    typer.run(main)
