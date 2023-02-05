from pprint import pprint
import sys
from datasynth.pipelines import AddressTestPipeline, NameTestPipeline
from datasynth.generators import AddressGenerator, NameGenerator
from datasynth.normalizers import AddressNormalizer, NameNormalizer
import typer

pipelines = {"address": AddressTestPipeline, "name": NameTestPipeline}
generators = {"address": AddressGenerator, "name": NameGenerator}


def main(datatype: str = "address"):
    chain = generators[sys.argv[1]]()
    # No-op thing is a hack, not sure why it won't let me run with no args
    pprint(chain.run(noop="true"))


if __name__ == "__main__":
    typer.run(main)
