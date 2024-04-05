from typing import ClassVar
from pydantic import Field, PrivateAttr
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import typer
from pprint import pprint
from fuxion.base import BaseChain
from fuxion.few_shot import populate_few_shot, generate_population
from fuxion.settings import SEPARATOR
from fuxion.models import get_model

separator = SEPARATOR


class GeneratorChain(BaseChain):
    model_name: str = "gpt-3.5-turbo"
    template: PromptTemplate = PrivateAttr()
    chain = Field(LLMChain, required=False)
    chain_type: ClassVar[str] = "generator"
    temperature: float = 0.0
    cache: bool = False
    verbose: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.template = PromptTemplate(
            input_variables=["few_shot"],
            template=open(self.template_file).read(),
            validate_template=False,
            template_format="jinja2",
        )
        if self.temperature > 2.0:
            raise ValueError(
                f"temperature:{self.temperature} is greater than the maximum of 2-'temperature'"
            )
        self.chain = LLMChain(
            prompt=self.template,
            llm=get_model(
                temperature=self.temperature,
                cache=self.cache,
                model_name=self.model_name,
            ),
            verbose=self.verbose,
        )

    def execute(
        self,
        few_shot_example_file: str,
        sample_size: int = 3,
    ):
        population = generate_population(few_shot_example_file=few_shot_example_file)
        few_shot = populate_few_shot(population=population, sample_size=sample_size)
        result = self.chain.invoke({"few_shot": few_shot})
        return result

    @classmethod
    def from_template(
        cls,
        *args,
        **kwargs,
    ):
        return super().from_name(
            *args,
            class_suffix="Generator",
            base_cls=GeneratorChain,
            chain_type="generator",
            **kwargs,
        )

    @property
    def input_keys(self) -> list[str]:
        return self.chain.input_keys

    @property
    def output_keys(self) -> list[str]:
        return ["generated"]

    def _call(self, inputs: dict[str, str]) -> dict[str, list[str]]:
        generated_items = self.chain.invoke(input=inputs)["text"].split(separator)
        filtered_items = [item for item in generated_items if item.strip()]
        return {"generated": filtered_items}


def main(
    generator_template: str,
    few_shot_example_file: str,
    sample_size: int = 3,
    temperature: float = 0.5,
    cache: bool = False,
    verbose: bool = True,
    model_name: str = "gpt-3.5-turbo",
):
    population = generate_population(few_shot_example_file=few_shot_example_file)
    few_shot = populate_few_shot(population=population, sample_size=sample_size)
    chain = GeneratorChain.from_template(
        generator_template,
        temperature=temperature,
        cache=cache,
        verbose=verbose,
        model_name=model_name,
    )
    pprint(chain.invoke(few_shot))


if __name__ == "__main__":
    typer.run(main)
