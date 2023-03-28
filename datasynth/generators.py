import os
from typing import ClassVar
from pydantic import Field, PrivateAttr
from langchain import LLMChain
from langchain import OpenAI
from langchain import PromptTemplate
import typer
from datasynth import TEMPLATE_DIR
from pprint import pprint
from datasynth.base import BaseChain, auto_class
import typing
import time
from few_shot import populate_few_shot, generate_population
# davinci = OpenAI(
#     temperature=0.8,
#     cache=True,
# )


class GeneratorChain(BaseChain):
    _template: PromptTemplate = PrivateAttr()
    chain = Field(LLMChain, required=False)
    chain_type: ClassVar[str] = "generator"
    temperature: float = 0.0
    cache: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._template = PromptTemplate(
            input_variables=["few_shot"],
            template=open(
                os.path.join(TEMPLATE_DIR, "generator", f"{self.datatype}.template")
            ).read(),
            validate_template=False,
            template_format="jinja2",
        )

        if self.temperature > 2.0:
            raise ValueError(
                f"temperature:{self.temperature} is greater than the maximum of 2-'temperature'"
            )
        self.chain = LLMChain(
                    prompt=self._template,
                    llm=OpenAI(temperature=self.temperature, cache=self.cache),
                    verbose=True,
                )

    @property
    def input_keys(self) -> list[str]:
        return self.chain.input_keys

    @property
    def output_keys(self) -> list[str]:
        return ["generated"]

    def _call(self, inputs: dict[str, str]) -> dict[str, list[str]]:
        generated_items = self.chain.run(**inputs).split("\n\n")
        filtered_items = [item for item in generated_items if item.strip()]
        return {"generated": filtered_items}


template_dir = os.path.join(TEMPLATE_DIR, "generator")
auto_class(template_dir, GeneratorChain, "Generator")


def main(datatype: str, sample_size: int = 3,  temperature: float = 0.5, cache: bool = False):
    population = generate_population(datatype)
    few_shot = populate_few_shot(population=population, sample_size=sample_size)
    chain = GeneratorChain.from_name(datatype, temperature=temperature, cache=cache)
    # No-op thing is a hack, not sure why it won't let me run with no args
    pprint(chain.run(few_shot=few_shot, noop="true"))


if __name__ == "__main__":
    typer.run(main)
