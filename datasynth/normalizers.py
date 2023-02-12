import json
import ast
from pprint import pprint
import os
import typer
from pydantic import Field
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.cache import SQLiteCache
from langchain.chains.base import Chain
import langchain
from datasynth.base import BaseChain, auto_class
from datasynth import TEMPLATE_DIR

langchain.llm_cache = SQLiteCache()

structured_davinci = OpenAI(
    temperature=0,
    cache=True,
    stop=["]"],
)


class NormalizerChain(BaseChain):
    chain = Field(LLMChain, required=False)
    chain_type = "normalizer"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        template = PromptTemplate(
            input_variables=[self.datatype],
            template=open(
                os.path.join(TEMPLATE_DIR, "normalizer", f"{self.datatype}.template")
            ).read(),
            validate_template=True,
            template_format="jinja2",
        )
        self.chain = LLMChain(
            prompt=template,
            llm=structured_davinci,
            verbose=True,
        )

    @property
    def input_keys(self) -> list[str]:
        return self.chain.input_keys

    @property
    def output_keys(self) -> list[str]:
        return ["normalized"]

    def _call(self, inputs: dict[str, str]) -> dict[str, list]:
        output = "[{" + self.chain.run(**inputs) + "]"
        try:
            return {"normalized": ast.literal_eval(output)}
        except json.JSONDecodeError:
            print("Failed to normalize", output)
            return {"normalized": []}


template_dir = os.path.join(TEMPLATE_DIR, "normalizer")
auto_class(template_dir, NormalizerChain, "Normalizer")


def main(datatype: str, example:str):

    chain = NormalizerChain.from_name(datatype)
    pprint(chain.run(**{datatype: example}))
    

if __name__ == "__main__":
    typer.run(main)