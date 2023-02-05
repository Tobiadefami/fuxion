from pprint import pprint
import sys
from datasynth.pipelines import AddressTestPipeline, NameTestPipeline, PriceTestPipeline
from datasynth.generators import AddressGenerator, NameGenerator, PriceGenerator
from datasynth.normalizers import AddressNormalizer, NameNormalizer
import typer

pipelines = {"address": AddressTestPipeline, "name": NameTestPipeline, "price":PriceTestPipeline}
generators = {"address": AddressGenerator, "name": NameGenerator, "price": PriceGenerator}


def main(datatype: str = "address"):
    chain = pipelines[datatype]()
    # No-op thing is a hack, not sure why it won't let me run with no args
    pprint(chain.run(noop="true"))


if __name__ == "__main__":
    typer.run(main)
