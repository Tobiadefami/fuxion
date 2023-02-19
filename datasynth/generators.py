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
            input_variables=[],
            template=open(
                os.path.join(TEMPLATE_DIR, "generator", f"{self.datatype}.template")
            ).read(),
            validate_template=False,
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
        return {"generated": generated_items}


template_dir = os.path.join(TEMPLATE_DIR, "generator")
auto_class(template_dir, GeneratorChain, "Generator")


def main(datatype: str, temperature: float = 0.5, cache: bool = False):
    chain = GeneratorChain.from_name(datatype, temperature=temperature, cache=cache)
    # No-op thing is a hack, not sure why it won't let me run with no args
    pprint(chain.run(noop="true"))


if __name__ == "__main__":
    typer.run(main)
