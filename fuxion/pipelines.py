import typer
import time
import glob
from typing import Optional, Dict, Any, List
from fuxion.generators import GeneratorChain
from fuxion.few_shot import generate_population, populate_few_shot
import os
import json
from tqdm import tqdm

class DatasetPipeline:
    def __init__(
        self,
        generator_template: str,
        few_shot_file: str,
        output_structure: Dict[str, Any],
        k: int = 10,
        sample_size: int = 3,
        batch_save: bool = False,
        batch_size: int = 100,
        dataset_name: Optional[str] = None,
        manual_review: bool = False,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        verbose: bool = False,
        cache: bool = False
    ):
        self.generator = GeneratorChain(
            template_file=generator_template,
            output_structure=output_structure,
            model_name=model_name,
            temperature=temperature,
            verbose=verbose,
            cache=cache
        )
        self.few_shot_file = few_shot_file
        self.k = k
        self.sample_size = sample_size
        self.batch_save = batch_save
        self.batch_size = batch_size
        self.dataset_name = dataset_name or str(int(time.time()))
        self.manual_review = manual_review

    def execute(self) -> Dict[str, List[Dict[str, Any]]]:
        population = generate_population(few_shot_example_file=self.few_shot_file)
        outputs = []
        pbar = tqdm(total=self.k, desc="Generating Dataset",  unit="items")
        while len(outputs) < self.k:
            few_shot = populate_few_shot(population=population, sample_size=self.sample_size)
            generated_content = self.generator.generate(few_shot)

            for item in generated_content.items:
                if len(outputs) < self.k:
                    output = {
                        "raw": str(item),
                        "structured": item.dict()
                    }
                    outputs.append(output)
                    pbar.update(1)
                # if len(outputs) >= self.k:
                else:
                    break


            if self.batch_save and len(outputs) % self.batch_size == 0:
                self._save_batch(outputs[-self.batch_size:], len(outputs) // self.batch_size)
        pbar.close()
        results = {
            "dataset": {
                "outputs": outputs[:self.k],
                "generator_prompt": self.generator.template.template,
            }
        }

        if self.manual_review:
            for item in results["dataset"]["outputs"]:
                print(f"Generated Item: {item}\n--------")
                user_input = input("Accept? (y/N): ")
                item["manual_review"] = "accepted" if user_input.lower() == "y" else "rejected"

        if not self.batch_save:
            self._save_full_dataset(results)

        return results["dataset"]

    def _save_batch(self, batch: List[Dict[str, Any]], batch_index: int):
        batch_name = f"{self.dataset_name}_batch_{batch_index}.json"
        self._save_json(batch_name, {"dataset": {"outputs": batch, "generator_prompt": self.generator.template.template}})

    def _save_full_dataset(self, results: Dict[str, Dict[str, List[Dict[str, Any]] | str]]):
        self._save_json(f"{self.dataset_name}.json", results)

    def _save_json(self, filename: str, data: Dict):
        dataset_dir = os.path.join(os.path.dirname(__file__), "datasets")
        os.makedirs(dataset_dir, exist_ok=True)
        file_path = os.path.join(dataset_dir, filename)
        with open(file_path, "w") as fd:
            json.dump(data, fd)

def generate_dataset(
    generator_file: str,
    example_file: str,
    output_structure: str,
    k: int = 5,
    dataset_name: Optional[str] = None,
    temperature: float = 0.8,
    manual_review: bool = False,
    model_name: str = "gpt-3.5-turbo",
    batch_save: bool = False,
    batch_size: int = 10,
    verbose: bool = False,
    cache: bool = False
):
    import ast
    structure_dict = ast.literal_eval(output_structure)

    pipeline = DatasetPipeline(
        generator_template=generator_file,
        few_shot_file=example_file,
        output_structure=structure_dict,
        k=k,
        temperature=temperature,
        model_name=model_name,
        batch_save=batch_save,
        batch_size=batch_size,
        dataset_name=dataset_name,
        manual_review=manual_review,
        verbose=verbose,
        cache=cache
    )

    return pipeline.execute()

if __name__ == "__main__":
    typer.run(generate_dataset)
