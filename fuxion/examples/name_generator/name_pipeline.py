from fuxion.pipelines import DatasetPipeline


pipeline_chain = DatasetPipeline.from_template(
    generator_template="examples/name_generator/generator.template",
    normalizer_template="examples/name_generator/normalizer.template",
    few_shot_file="examples/name_generator/few_shot.json",
    dataset_name="name_pipeline",
    k=20,
    model_name="gpt-3.5-turbo",
    cache=True,
    verbose=True,
    temperature=1.0,
    batch_save=True,
    batch_size = 3,
)

result = pipeline_chain.execute()
print(result)