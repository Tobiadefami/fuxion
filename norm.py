import json
import ast
from pprint import pprint

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.cache import SQLiteCache
from langchain.chains.base import Chain
import langchain

langchain.llm_cache = SQLiteCache()

structured_davinci = OpenAI(
    temperature=0,
    cache=True,
    stop=["]"],
)
davinci = OpenAI(
    temperature=0.8,
    cache=True,
)


class NormalizerChain(Chain):
    chain: LLMChain

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


class GeneratorChain(Chain):
    chain: LLMChain

    @property
    def input_keys(self) -> list[str]:
        return self.chain.input_keys

    @property
    def output_keys(self) -> list[str]:
        return ["generated"]

    def _call(self, inputs: dict[str, str]) -> dict[str, list[str]]:
        generated_items = self.chain.run(**inputs).split("\n\n")
        return {"generated": generated_items}


class TestPipeline(Chain):
    generator: GeneratorChain
    discriminator: NormalizerChain

    @property
    def input_keys(self) -> list[str]:
        return self.generator.input_keys

    @property
    def output_keys(self) -> list[str]:
        return ["outputs"]

    def _call(self, inputs: dict[str, str]) -> dict[str, list]:
        generated = self.generator.run(**inputs)
        return {
            "outputs": [
                {
                    "input": example,
                    "output": self.discriminator.run(
                        **{self.discriminator.input_keys[0]: example}
                    ),
                }
                for example in generated
            ]
        }


name_norm_prompt = PromptTemplate(
    input_variables=["name"],
    template=open("name.template").read(),
    validate_template=True,
    template_format="jinja2",
)


class NameNormalizer(NormalizerChain):
    chain = LLMChain(prompt=name_norm_prompt, llm=structured_davinci, verbose=True)


address_norm_prompt = PromptTemplate(
    input_variables=["address"],
    template=open("address.template").read(),
    validate_template=True,
    template_format="jinja2",
)


class AddressNormalizer(NormalizerChain):
    chain = LLMChain(prompt=address_norm_prompt, llm=structured_davinci, verbose=True)


address_generator_prompt = PromptTemplate(
    input_variables=[],
    template=open("address.generator.template").read(),
    validate_template=False,
)


class AddressGenerator(GeneratorChain):
    chain = LLMChain(prompt=address_generator_prompt, llm=davinci)


class AddressTestPipeline(TestPipeline):
    generator = AddressGenerator()
    discriminator = AddressNormalizer()


chain = AddressTestPipeline()

# No-op thing is a hack, not sure why it won't let me run with no args
pprint(chain.run(noop="true"))
