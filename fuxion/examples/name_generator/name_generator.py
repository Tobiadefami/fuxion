from fuxion.generators import GeneratorChain
from pprint import pprint

chain = GeneratorChain.from_template(
    template_file="examples/name_generator/generator.template",
    temperature=0.0,
    cache=True,
    verbose=True,
    model_name="gpt-3.5-turbo",
)


result = chain.execute(
    few_shot_example_file="examples/name_generator/few_shot.json", sample_size=3
)
pprint(result)
