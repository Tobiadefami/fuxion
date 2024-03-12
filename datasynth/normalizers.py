import json
import ast
from pprint import pprint
import os
import typer
from pydantic import Field, PrivateAttr
from datasynth.base import BaseChain, auto_class
from datasynth import TEMPLATE_DIR
from typing import Any
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from models import get_model

optional_params = {"stop": ["]"]}


class NormalizerChain(BaseChain):
    model_name: str = "gpt-3.5-turbo"
    _template: PromptTemplate = PrivateAttr()
    chain = Field(LLMChain, required=False)
    chain_type = "normalizer"
    temperature: float = 0.0
    cache: bool = True
    verbose: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._template = PromptTemplate(
            input_variables=[self.datatype],
            template=open(
                os.path.join(TEMPLATE_DIR, "normalizer", f"{self.datatype}.template")
            ).read(),
            validate_template=True,
            template_format="jinja2",
        )
        if self.temperature > 2.0:
            raise ValueError(
                f"temperature:{self.temperature} is greater than the maximum of 2-'temperature'"
            )
        self.chain = LLMChain(
            prompt=self._template,
            llm=get_model(
                model_name=self.model_name,
                temperature=self.temperature,
                cache=self.cache,
                model_kwargs=optional_params,
            ),
            verbose=self.verbose,
        )

    @property
    def input_keys(self) -> list[str]:
        return self.chain.input_keys

    @property
    def output_keys(self) -> list[str]:
        return ["normalized"]

    def _call(self, inputs: dict[str, str]) -> dict[str, list[Any]]:
        print("input >>", inputs)
        output = "[{" + self.chain.invoke(inputs)["text"] + "]"
        print("OUTPUTS >>", output)
        try:
            return {"normalized": ast.literal_eval(output)}
        except json.JSONDecodeError:
            print("Failed to normalize", output)
            return {"normalized": []}


template_dir = os.path.join(TEMPLATE_DIR, "normalizer")
auto_class(template_dir, NormalizerChain, "Normalizer")


def main(
    datatype: str,
    example: str,
    temperature: float = 0.0,
    cache: bool = False,
    verbose: bool = True,
    model_name: str = "gpt-3.5-turbo",
):

    chain = NormalizerChain.from_name(
        datatype,
        temperature=temperature,
        cache=cache,
        verbose=verbose,
        model_name=model_name,
    )
    inputs = {datatype: example}
    pprint(chain.invoke(inputs))


if __name__ == "__main__":
    typer.run(main)
