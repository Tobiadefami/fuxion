from datasynth.generators import GeneratorChain

generator = GeneratorChain.execute(
    generator_template="examples/name_generator/generator.template",
    few_shot_example_file="examples/name_generator/few_shot.json",
    sample_size=3,
    temperature=0.5,
    cache=False,
    verbose=True,
    model_name='gpt-3.5-turbo',
)

print(generator)