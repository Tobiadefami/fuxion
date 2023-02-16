import typer
import time
from typing import Optional
from datasynth.generators import *
from datasynth.normalizers import *
from typing import Any, List, ClassVar, Optional
from datasynth.base import BaseChain
from typing import ClassVar

# TODO: Figure out how to specify number of example we want out and run until that many examples are generated.  Make sure we save the output to a JSON file so we persist it.


class TestPipeline(BaseChain):
    k: Optional[int] = 10
    dataset_name: Optional[str] = None
    # manual_review: bool = False
    generator: ClassVar[GeneratorChain]
    normalizer: ClassVar[NormalizerChain]
    chain_type = "testpipeline"

    # def __init__(self, *args, **kwargs):
    #     self.k = kwargs.pop('k', 10)
    #     super().__init__(*args, **kwargs)

    @property
    def input_keys(self) -> List[str]:
        return self.generator.input_keys

    @property
    def output_keys(self) -> List[str]:
        return ["dataset"]

    def _call(
        self,
        inputs: dict[str, str],
    ) -> dict[str, List[dict[str, Any | str]]]:
        generated: List[dict[str, Any | str]] = []
        while len(generated) < self.k:
            generated.extend(self.generator.run(**inputs))

        results: dict[str, dict[str, list[dict[str, Any|str]] | str]] = {
            "dataset": {
                "outputs": [
                    {
                        "input": example,
                        "output": self.normalizer.run(
                            **{self.normalizer.input_keys[0]: example}
                        ),
                    }
                    for example in generated[: self.k]
                ],
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

        dataset_path = os.path.join("datasets", self.dataset_name)
        with open(dataset_path, "w") as fd:
            json.dump(results, fd)

        return results


# TODO: look at overlap of generator and normalizer dirs
template_dir = os.path.join(TEMPLATE_DIR, "generator")
auto_class(
    template_dir,
    TestPipeline,
    "TestPipeline",
    generator_chain=GeneratorChain,
    normalizer_chain=NormalizerChain,
)


def main(datatype: str, k: int = 10, dataset_name: str = None):
    chain = TestPipeline.from_name(datatype, k=k, dataset_name=dataset_name)
    # No-op thing is a hack, not sure why it won't let me run with no args
    pprint(chain.run(noop="true"))


if __name__ == "__main__":
    typer.run(main)