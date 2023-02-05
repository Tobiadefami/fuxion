from langchain.chains.base import Chain

from datasynth.generators import *
from datasynth.normalizers import *


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


class AddressTestPipeline(TestPipeline):
    generator = AddressGenerator()
    discriminator = AddressNormalizer()


class NameTestPipeline(TestPipeline):
    generator = NameGenerator()
    discriminator = NameNormalizer()
