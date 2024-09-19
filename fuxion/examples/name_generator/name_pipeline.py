from fuxion.pipelines import DatasetPipeline
from rich import print

output_structure = {
    "title": str ,
    "given": str,
    "middle": str ,
    "surname": str,
    "suffix": str
}

pipeline_chain = DatasetPipeline(
    generator_template="examples/name_generator/generator.template",
    few_shot_file="examples/name_generator/few_shot.json",
    output_structure=output_structure,
    dataset_name="name_pipeline",
    k=5,
    model_name="gpt-4o",
    cache=False,
    verbose=True,
    temperature=1.0,
)

result = pipeline_chain.execute()
print(result)
