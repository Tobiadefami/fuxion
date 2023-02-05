from pprint import pprint
import sys
from datasynth.pipelines import AddressTestPipeline, NameTestPipeline

pipelines = {"address": AddressTestPipeline, "name": NameTestPipeline}


if __name__ == "__main__":
    chain = pipelines[sys.argv[1]]()

    # No-op thing is a hack, not sure why it won't let me run with no args
    pprint(chain.run(noop="true"))
