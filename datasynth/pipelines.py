from langchain.chains.base import Chain
import typer
from datasynth.generators import *
from datasynth.normalizers import *

#TODO: Figure out how to specify number of example we want out and run until that many examples are generated.  Make sure we save the output to a JSON file so we persist it.


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
        generated = []
        while len(generated) < 30:
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
    generator = AddressGenerator()
    discriminator = AddressNormalizer()


class NameTestPipeline(TestPipeline):
    generator = NameGenerator()
    discriminator = NameNormalizer()

class PriceTestPipeline(TestPipeline):
    generator = PriceGenerator()
    discriminator = PriceNormalizer()


def main(datatype: str):
    registry = {
    'name': NameTestPipeline,
    'address': AddressTestPipeline,
    'price': PriceTestPipeline}
    
    chain = registry[datatype]()
    # No-op thing is a hack, not sure why it won't let me run with no args
    pprint(chain.run(noop="true"))



if __name__ == "__main__":
    typer.run(main)