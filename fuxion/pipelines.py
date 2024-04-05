import typer
import time
import glob
from typing import Optional, Any
from fuxion.generators import *
from fuxion.normalizers import *
from typing import Any, List, ClassVar, Optional
from fuxion.base import BaseChain
from typing import ClassVar
from fuxion.few_shot import generate_population, populate_few_shot
import os
from fuxion.generators import GeneratorChain
from fuxion.normalizers import NormalizerChain


class DatasetPipeline(BaseChain):

    k: int = 10
    few_shot_file: str
    sample_size: int = 3
    batch_save: bool = False
    batch_size: int = 100
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

    def execute(
        self,
    ) -> dict[str, List[dict[str, Any | str]]]:

        return self.run()

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

    def save_batch(self, batch: list[dict[str, Any]], batch_index: int):
        """Save a batch of outputs to a file, with each batch saved in a separate file."""

        if self.dataset_name is None:
            self.dataset_name = str(int(time.time()))

        batch_name = f"{self.dataset_name}_batch_{batch_index}.json"
        batch_dir = os.path.join(os.path.dirname(__file__), "datasets")
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir, exist_ok=True)

        batch_path = os.path.join(batch_dir, batch_name)
        batch_outputs: dict[str, dict[str, list[dict[str, Any | str]] | str]] = {
            "dataset": {
                "outputs": batch,
                "generator_prompt": self.generator.template.template,
                "normalizer_prompt": self.normalizer.template.template,
            }
        }
        with open(batch_path, "w") as fd:
            json.dump(batch_outputs, fd)

    def save_full_dataset(
        self, results: dict[str, dict[str, list[dict[str, Any | str]] | str]]
    ) -> None:
        """Save the complete dataset to a file."""
        if self.dataset_name is None:
            self.dataset_name = f"{str(int(time.time()))}.json"
        else:
            self.dataset_name = f"{self.dataset_name}.json"

        dataset_dir = os.path.join(os.path.dirname(__file__), "datasets")

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)

        dataset_path = os.path.join(dataset_dir, self.dataset_name)

        with open(dataset_path, "w") as fd:
            json.dump(results, fd)

    def load_from_checkpoint(self) -> tuple[list[dict[str, Any | str]], int]:

        if self.dataset_name is None:
            self.dataset_name = str(int(time.time()))

        checkpoint_dir = os.path.join(os.path.dirname(__file__), "datasets")
        pattern = os.path.join(checkpoint_dir, f"{self.dataset_name}_batch_*.json")
        batch_files = sorted(
            glob.glob(pattern), key=lambda x: int(x.split("_")[-1].split(".")[0])
        )

        if not os.path.exists(os.path.join(checkpoint_dir, self.dataset_name)):
            return [], 0

        existing_outputs: list[dict[str, Any]] = []
        for batch_file in batch_files:
            with open(batch_file, "r") as fd:
                batch_data: list[dict[str, Any]] = json.load(fd)
                existing_outputs.extend(batch_data)

        # If there are batch files, set the next batch index to one higher than the last found batch index.
        if batch_files:
            last_batch_index = int(
                batch_files[-1].split("_")[-1].split(".")[0]
            )  # Assumes naming: datasetName_batch_N.json
            batch_index = last_batch_index + 1
        else:
            batch_index = 0

        return existing_outputs, batch_index

    def _call(
        self,
        inputs: Optional[dict[str, str] | None] = None,
    ) -> dict[str, List[dict[str, Any | str]]]:

        population = generate_population(few_shot_example_file=self.few_shot_file)

        batch_outputs: list[dict[str, Any]] = []
        outputs, batch_idx = self.load_from_checkpoint()

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
                    normalized_output = self.normalizer.invoke(normalizer_inputs)
                    output = {
                        "generated_input": example,
                        "normalized_output": normalized_output,
                    }
                    outputs.append(output)

                    if self.batch_save:
                        # Append the current output to the batch_outputs list
                        batch_outputs.append(output)
                        if len(batch_outputs) >= self.batch_size:
                            # Save the current batch to a file
                            self.save_batch(batch_outputs, batch_idx)
                            # Reset the batch_outputs list
                            batch_outputs = []
                            # Increment the batch index for the next batch
                            batch_idx += 1

                    if len(outputs) >= self.k:
                        break

                except Exception as e:
                    print(e)
                    continue

        # save any reminaing items to last batch
        if batch_outputs and self.batch_save:
            self.save_batch(batch_outputs, batch_idx)

        results: dict[str, dict[str, list[dict[str, Any | str]] | str]] = {
            "dataset": {
                "outputs": outputs,
                "generator_prompt": self.generator.template.template,
                "normalizer_prompt": self.normalizer.template.template,
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

        if not self.batch_save:
            self.save_full_dataset(results)

        return results["dataset"]


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
    batch_save: bool = False,
    batch_size: int = 10,
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
        batch_save=batch_save,
        batch_size=batch_size,
        dataset_name=dataset_name,
        manual_review=manual_review,
    )

    return chain.run()


if __name__ == "__main__":
    typer.run(generate_dataset)
