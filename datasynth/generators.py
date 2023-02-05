import os

from pydantic import Field
from langchain import LLMChain
from langchain.chains.base import Chain
from langchain import OpenAI
from langchain import PromptTemplate

from datasynth import TEMPLATE_DIR

davinci = OpenAI(
    temperature=0.8,
    cache=False,
)


class GeneratorChain(Chain):
    datatype: str
    chain = Field(LLMChain, required=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        template = PromptTemplate(
            input_variables=[],
            template=open(
                os.path.join(TEMPLATE_DIR, f"{self.datatype}.generator.template")
            ).read(),
            validate_template=False,
        )
        self.chain = LLMChain(prompt=template, llm=davinci, verbose=True)

    @property
    def input_keys(self) -> list[str]:
        return self.chain.input_keys

    @property
    def output_keys(self) -> list[str]:
        return ["generated"]

    def _call(self, inputs: dict[str, str]) -> dict[str, list[str]]:
        generated_items = self.chain.run(**inputs).split("\n\n")
        return {"generated": generated_items}


class AddressGenerator(GeneratorChain):
    datatype = "address"


class NameGenerator(GeneratorChain):
    datatype = "name"


class PriceGenerator(GeneratorChain):
    datatype = "price"