import json
import ast
import re
from pprint import pprint
import typer
from pydantic import Field, PrivateAttr
from datasynth.base import BaseChain
from typing import Any, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from datasynth.models import get_model

optional_params = {"stop": ["]"]}


class NormalizerChain(BaseChain):
    model_name: str = "gpt-3.5-turbo"
    _template: PromptTemplate = PrivateAttr()
    chain = Field(LLMChain, required=False)
    chain_type = "normalizer"
    temperature: float = 0.0
    cache: bool = True
    verbose: bool = True
    datatype: Optional[str]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with open(self.template_file) as f:
            data = f.read()
            pattern = r"\{\{(\w+)\}\}"
            match = re.search(pattern, data)
            self.datatype = match.group(1)

        self._template = PromptTemplate(
            input_variables=[self.datatype],
            template=open(self.template_file).read(),
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

    @staticmethod
    def execute(
        normalizer_template: str,
        example: str,
        temperature: float = 0.0,
        cache: bool = False,
        verbose: bool = True,
        model_name: str = "gpt-3.5-turbo",
    ):
        chain = NormalizerChain.from_template(
            template_file=normalizer_template,
            temperature=temperature,
            cache=cache,
            verbose=verbose,
            model_name=model_name,
        )
        result = chain.invoke({chain.datatype: example})
        pprint(result)
        
    @classmethod
    def from_template(
        cls,
        *args,
        **kwargs,
    ):
        return super().from_name(
            *args,
            class_suffix="Normalizer",
            base_cls=NormalizerChain,
            chain_type="normalizer",
            **kwargs,
        )

    @property
    def input_keys(self) -> list[str]:
        return self.chain.input_keys

    @property
    def output_keys(self) -> list[str]:
        return ["normalized"]

    def _call(self, inputs: dict[str, str]) -> dict[str, list[Any]]:
        output = "[{" + self.chain.invoke(inputs)["text"] + "]"

        try:
            return {"normalized": ast.literal_eval(output)}
        except json.JSONDecodeError:
            print("Failed to normalize", output)
            return {"normalized": []}


def main(
    normalizer_template: str,
    example: str,
    temperature: float = 0.0,
    cache: bool = False,
    verbose: bool = True,
    model_name: str = "gpt-3.5-turbo",
):

    chain = NormalizerChain.from_template(
        normalizer_template,
        temperature=temperature,
        cache=cache,
        verbose=verbose,
        model_name=model_name,
    )
    pprint(chain.invoke({chain.datatype:example}))


if __name__ == "__main__":
    typer.run(main)
