from datasynth.pipelines import DatasetPipeline

# Create a new pipeline
pipeline = DatasetPipeline.execute(
    generator_file="examples/name_generator/generator.template",
    normalizer_file="examples/name_generator/normalizer.template",
    few_shot_example_file="examples/name_generator/few_shot.json",
    dataset_name="name_pipeline"
)


print(pipeline)