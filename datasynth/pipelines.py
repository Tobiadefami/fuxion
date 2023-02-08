from langchain.chains.base import Chain
import typer
from datasynth.generators import *
from datasynth.normalizers import *
from typing import Any, List
from datasynth.base import BaseChain
#TODO: Figure out how to specify number of example we want out and run until that many examples are generated.  Make sure we save the output to a JSON file so we persist it.


class TestPipeline(BaseChain):
    generator: GeneratorChain
    discriminator: NormalizerChain
    chain_type = "testpipeline"

    @property
    def input_keys(self) -> List[str]:
        return self.generator.input_keys

    @property
    def output_keys(self) -> List[str]:
        return ["outputs"]

    def _call(self, inputs: dict[str, str], k:int = 30) -> dict[str, List[dict[str, Any | str]]]:
        generated: List[dict[str, Any | str]] = []
        while len(generated) < k:
            generated.extend(self.generator.run(**inputs))
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


class AddressTestPipeline(TestPipeline):
    datatype = "address"
    generator = AddressGenerator()
    discriminator = AddressNormalizer()


class NameTestPipeline(TestPipeline):
    datatype = "name"
    generator = NameGenerator()
    discriminator = NameNormalizer()

class PriceTestPipeline(TestPipeline):
    datatype = "price"
    generator = PriceGenerator()
    discriminator = PriceNormalizer()


def main(datatype: str):
    
    chain = TestPipeline.from_name(datatype)
    # No-op thing is a hack, not sure why it won't let me run with no args
    pprint(chain.run(noop="true"))



if __name__ == "__main__":
    typer.run(main)