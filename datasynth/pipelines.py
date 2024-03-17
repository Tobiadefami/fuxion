import typer
import time
from typing import Optional, Any
from datasynth.generators import *
from datasynth.normalizers import *
from typing import Any, List, ClassVar, Optional
from datasynth.base import BaseChain
from typing import ClassVar
from datasynth.few_shot import generate_population, populate_few_shot
import os
from datasynth.generators import GeneratorChain
from datasynth.normalizers import NormalizerChain


class DatasetPipeline(BaseChain):

    k: int = 10
    few_shot_file: str
    sample_size: int = 3
    dataset_name: Optional[str] = None
    manual_review: bool = False
    generator: ClassVar[GeneratorChain]
    normalizer: ClassVar[NormalizerChain]
    chain_type = "DatasetPipeline"

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return ["dataset"]

    @classmethod
    def from_template(
        cls,
        *args,
        **kwargs,
    ):
        return super()._from_name(
            *args,
            generator_chain=GeneratorChain,
            normalizer_chain=NormalizerChain,
            class_suffix="DatasetPipeline",
            base_cls=DatasetPipeline,
            **kwargs,
        )
        
    @staticmethod
    def execute(
        generator_file: str,
        normalizer_file: str,
        few_shot_example_file: str,
        k: int = 5,
        dataset_name: Optional[str] = None,
        temperature: float = 0.8,
        cache: bool = False,
        manual_review: bool = False,
        model_name: str = "gpt-3.5-turbo",
    ) -> dict[str, List[dict[str, Any | str]]]:
        """
        Static method to generate a dataset using the DatasetPipeline. This
        method initializes the pipeline with given templates and parameters,
        then invokes the pipeline to generate and normalize a dataset.

        Args:
            generator_file: Path to the generator template file.
            normalizer_file: Path to the normalizer template file.
            example_file: Path to the few-shot example file.
            k: Number of samples to generate.
            dataset_name: Custom name for the dataset file.
            temperature: Model temperature for generation.
            cache: Whether to cache model calls.
            manual_review: Whether to enable manual review of generated data.
            model_name: Name of the model to use.

        Returns:
            A dictionary containing the generated and normalized dataset.
        """
        chain = DatasetPipeline.from_template(
            generator_template=generator_file,
            normalizer_template=normalizer_file,
            few_shot_file=few_shot_example_file,
            k=k,
            temperature=temperature,
            cache=cache,
            model_name=model_name,
            dataset_name=dataset_name,
            manual_review=manual_review,
        )

        return chain.run()

    def run(
        self, inputs: dict[str, str] | None = None
    ) -> dict[str, List[dict[str, Any | str]]]:
        """
        Executes the dataset generation pipeline. This method is the primary entry point for running
        the dataset generation and normalization process. It delegates to the _call method with the
        provided inputs, ensuring that inputs are always in the expected format for the generator and
        normalizer chains.

        Args:
            inputs (Optional[dict[str, str]]): Optional dictionary of inputs to pass to the generator. Defaults to None.

        Returns:
            dict[str, List[dict[str, Any | str]]]: The results of the dataset generation and normalization process.
        """

        return self._call(inputs)

    def _call(
        self,
        inputs: Optional[dict[str, str] | None] = None,
    ) -> dict[str, List[dict[str, Any | str]]]:

        population = generate_population(few_shot_example_file=self.few_shot_file)

        outputs: list[dict[str, Any]] = []

        while len(outputs) < self.k:
            few_shot = populate_few_shot(
                population=population, sample_size=self.sample_size
            )

            inputs = inputs or {}
            inputs["few_shot"] = few_shot
            generated_content: dict[str, str] = self.generator.invoke(input=inputs)

            for example in generated_content["generated"]:

                try:
                    normalizer_inputs: dict[str, Any] = {
                        self.normalizer.input_keys[0]: example
                    }
                    outputs.append(
                        {
                            "generated_input": example,
                            "normalized_output": self.normalizer.invoke(
                                normalizer_inputs
                            ),
                        }
                    )
                    if len(outputs) >= self.k:
                        break
                except:
                    continue
        results: dict[str, dict[str, list[dict[str, Any | str]] | str]] = {
            "dataset": {
                "outputs": outputs,
                "generator_prompt": self.generator._template.template,
                "normalizer_prompt": self.normalizer._template.template,
            }
        }

        if self.manual_review:
            for pair in results["dataset"]["outputs"]:

                print(
                    f"{'Generated Input:', pair['generated_input']}\n-------\n{'Normalized Output:', pair['normalized_output']['normalized']}\n--------"
                )
                user_input = input("Accept? (y/N): ")
                if user_input.lower() == "y":
                    pair["manual_review"] = "accepted"
                else:
                    pair["manual_review"] = "rejected"

        if self.dataset_name is None:
            self.dataset_name = f"{str(int(time.time()))}.json"
        else:
            self.dataset_name = f"{self.dataset_name}.json"

        dataset_dir: str = os.path.join(os.path.dirname(__file__), "datasets")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)

        dataset_path: str = f"{dataset_dir}/{self.dataset_name}"
        with open(dataset_path, "w") as fd:
            json.dump(results, fd)

        return results


def generate_dataset(
    generator_file: str,
    normalizer_file: str,
    example_file: str,
    k: int = 5,
    dataset_name: str | None = None,
    temperature: float = 0.8,
    cache: bool = False,
    manual_review: bool = False,
    model_name: str = "gpt-3.5-turbo",
):
    """Generate synthetic data and the normalized output

    Args:
        datatype (str): Name of the data to be generated, given by the template's name.
        k (int, optional): Number of samples to generate | Defaults to 10.
        dataset_name (str, optional): Name of generated data (user defined) | Defaults to None.
        temperature (float, optional): Parameter that affects the randomness of the output | Defaults to 0.8.
        cache (bool, optional): Determine wheher to cache calls to/from the API | Defaults to False.

    Returns:
        dict[str, dict[str, list[dict[str, Any | str]] | str]] : A dictionary of synthetically generated and normalized data
    """

    chain = DatasetPipeline.from_template(
        generator_template=generator_file,
        normalizer_template=normalizer_file,
        few_shot_file=example_file,
        k=k,
        temperature=temperature,
        cache=cache,
        model_name=model_name,
        dataset_name=dataset_name,
        manual_review=manual_review,
    )

    return chain.run()


if __name__ == "__main__":
    typer.run(generate_dataset)
