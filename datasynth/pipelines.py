import typer
from datasynth.generators import *
from datasynth.normalizers import *
from typing import Any, List, ClassVar, Optional
from datasynth.base import BaseChain
#TODO: Figure out how to specify number of example we want out and run until that many examples are generated.  Make sure we save the output to a JSON file so we persist it.


class TestPipeline(BaseChain):
    """_summary_

    Args:
        BaseChain (_type_): _description_

    Returns:
        _type_: _description_
    """
    k: Optional[int] = 10
    generator: ClassVar[GeneratorChain]
    discriminator: ClassVar[NormalizerChain]
    chain_type = "testpipeline"
    
    # def __init__(self, *args, **kwargs):
    #     self.k = kwargs.pop('k', 10)
    #     super().__init__(*args, **kwargs)

        
    @property
    def input_keys(self) -> List[str]:
        return self.generator.input_keys

    @property
    def output_keys(self) -> List[str]:
        return ["outputs"]


    def _call(self, inputs: dict[str, str]) -> dict[str, List[dict[str, Any | str]]]:
        generated: List[dict[str, Any | str]] = []
        while len(generated) < self.k:
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
    
template_dir = os.path.join(TEMPLATE_DIR, "generator")
auto_class(template_dir, TestPipeline, "TestPipeline", generator_chain=GeneratorChain, normalizer_chain=NormalizerChain)


def main(datatype: str, k: int):
    
    chain = TestPipeline(k = k).from_name(datatype)
    # No-op thing is a hack, not sure why it won't let me run with no args
    pprint(chain.run(noop="true"))


if __name__ == "__main__":
    typer.run(main)