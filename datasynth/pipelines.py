import typer
import time
import random
from typing import Optional
from datasynth.generators import *
from datasynth.normalizers import *
from typing import Any, List, ClassVar, Optional
from datasynth.base import BaseChain
from typing import ClassVar
from few_shot import generate_population, populate_few_shot
# TODO: Figure out how to specify number of example we want out and run until that many examples are generated.  Make sure we save the output to a JSON file so we persist it.


class DatasetPipeline(BaseChain):
    k: int = 10
    sample_size: int = 3
    dataset_name: Optional[str] = None
    # manual_review: bool = False
    generator: ClassVar[GeneratorChain]
    normalizer: ClassVar[NormalizerChain]
    chain_type = "DatasetPipeline"

    # def __init__(self, *args, **kwargs):
    #     self.k = kwargs.pop('k', 10)
    #     super().__init__(*args, **kwargs)

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return ["dataset"]

    def _call(
        self,
        inputs: dict[str, str],
    ) -> dict[str, List[dict[str, Any | str]]]:
        generated: List[dict[str, Any | str]] = []

        population = generate_population(self.datatype)
        while len(generated) < self.k:
            few_shot = populate_few_shot(population=population, sample_size=self.sample_size)
            inputs["few_shot"] = few_shot
            
            generated.extend(self.generator.run(**inputs))

        outputs = []
        for example in generated[: self.k]:
            try:
                outputs.append(
                    {
                        "input": example,
                        "output": self.normalizer.run(
                            **{self.normalizer.input_keys[0]: example.strip()}
                        ),
                    }
                )
            except:
                continue

        results: dict[str, dict[str, list[dict[str, Any | str]] | str]] = {
            "dataset": {
                "outputs": outputs,
                "generator_prompt": self.generator._template.template,
                "normalizer_prompt": self.normalizer._template.template,
            }
        }

        # if self.manual_review:
        #     for pair in results["dataset"]["outputs"]:
        #         print(f"{pair['input']}\n-------\n{pair['output']}\n--------")
        #         user_input = input("Accept? (y/N): ")

        if self.dataset_name is None:
            self.dataset_name = f"{str(int(time.time()))}.json"
        else:
            self.dataset_name = f"{self.dataset_name}.json"

        dataset_path = os.path.join("datasets", self.dataset_name)
        with open(dataset_path, "w") as fd:
            json.dump(results, fd)

        return results


# TODO: look at overlap of generator and normalizer dirs
template_dir = os.path.join(TEMPLATE_DIR, "generator")
auto_class(
    template_dir,
    DatasetPipeline,
    "DatasetPipeline",
    generator_chain=GeneratorChain,
    normalizer_chain=NormalizerChain,
)


def generate_dataset(
    datatype: str,
    k: int = 10,
    dataset_name: str = None,
    temperature: float = 0.8,
    cache: bool = False,
):
    """ Generate

    Args:
        datatype (str): _description_
        k (int, optional): _description_. Defaults to 10.
        dataset_name (str, optional): _description_. Defaults to None.
        temperature (float, optional): _description_. Defaults to 0.8.
        cache (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    auto_class(
        template_dir,
        DatasetPipeline,
        "DatasetPipeline",
        generator_chain=GeneratorChain,
        normalizer_chain=NormalizerChain,
        temperature=temperature,
        cache=cache,
    )
    chain = DatasetPipeline.from_name(datatype, k=k, dataset_name=dataset_name)
    # No-op thing is a hack, not sure why it won't let me run with no args
    return chain.run(noop="true")


if __name__ == "__main__":
    typer.run(generate_dataset)
