# fuxion

LangChain + LLM powered data generation and normalization functions.
fuxion helps you generate a fully synthetic dataset with LLM APIs to train a task-specific model you can run on your own GPU.
Preliminary models for name, price, and address standardization are available on [HuggingFace](https://huggingface.co/PragmaticMachineLearning).

![fuxion](/assets/fuxion.png)

# Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
  - [Generation](#generation)
  - [Normalization](#normalization)
  - [Template Structure](#template-structure)
  - [Pipelines](#pipelines)

# Description

fuxion is a Python package that provides you with a data generation and normalization pipeline which could be used for testing, normalization and training machine learning models. Using fuxion, you are able to generate sythetic data for different types of use cases -- all that's required is that you pass the right prompt to the chain and watch how things unfold :sunglasses:

# Installation

We recommend that you create a virtual environment before proceeding with the installation process as it would help to create an isolated environment for this project. After doing that, you can proceed with the installation by following the steps below.

- install via pip

  ```bash
  pip install fuxion
  ```

- Add the following to your bashrc file and replace "your-key" with your OpenAI API key:

  ```bash
  export OPENAI_API_KEY="your-key"
  ```

# Usage

The process of creating useful synthetic data involves two main steps: data generation and normalization. fuxion provides a simple interface for both of these tasks, and a pipeline that chains together both of these tasks.

## Generation

```python

from fuxion.generators import GeneratorChain
from pprint import pprint

chain = GeneratorChain.from_template(
    template_file="examples/name_generator/generator.template",
    temperature=0.0,
    cache=False,
    verbose=True,
    model_name="gpt-3.5-turbo",
)


result = chain.execute(
    few_shot_example_file="examples/name_generator/few_shot.json", sample_size=3
)
pprint(result)
```

## Normalization

```python
from fuxion.normalizers import NormalizerChain

normalizer_chain = NormalizerChain.from_template(
    template_file="../templates/normalizer/address.template",
    temperature=0.0,
    cache=False,
    verbose=True,
    model_name="gpt-3.5-turbo",
)

normalizer = normalizer_chain.execute(
    example="John Doe street 1234, New York, NY 10001",
)
print(normalizer)

```

fuxion can be used to generate synthetic data for rapid product testing amongst other use cases. And this is easily achieved by passing the instructions and few shot examples as paths to the chain. The instructions are provided in the prompt template, and the few shot examples are provided in a json file.

## Template Structure

For each generation or normalization task, a template file is required to guide the llm on what to do. Below, we provide a brief overview of what the template files should look like for a given generation and normalization task.

##### Generator templates

```
Generate a list of U.S. postal addresses separated by double newlines.

Make them as realistic and diverse as possible.
Include some company address, P.O. boxes, apartment complexes, etc.
Ensure the addresses are fake.

{{few_shot}}

List:
```

- The first few lines tells the chain to generate addresses and contains a bunch of creative instructions that determines the quality of the results.

- `{{few_shot}}` tells the chain to get few-shot examples provided in the examples folder.

- `List` returns the results in a list

> The same convention should be followed when creating subsequent templates for various data generation tasks.

##### Normalizer templates

In the latest version of fuxion, normalization is integrated directly into the pipeline process using an `output_structure` parameter. This eliminates the need for separate normalization templates.

### Creating an Output Structure

The `output_structure` is a dictionary that defines the desired format for your normalized data. Here's how to create one:

```python
output_structure = {
    "field_name1": data_type,
    "field_name2": data_type,
    # ... more fields as needed
}
```

For example, for normalizing names:

```python
output_structure = {
    "title": str,
    "given": str,
    "middle": str,
    "surname": str,
    "suffix": str
}
```

Or for addresses:

```python
output_structure = {
    "house_number": int,
    "street": str,
    "city": str,
    "state": str,
    "zip_code": str
}
```

This structure guides the LLM in formatting the generated data, ensuring consistent and properly typed output. For details on how to use this in a pipeline, refer to the [Pipelines](#pipelines) section.

## Pipelines

We can train machine learning models on the combination of synthetically generated data and their normalized format. This is where we use `pipelines`.

The latest version of fuxion simplifies the normalization process by incorporating it directly into the pipeline using structured output. This removes the need for a separate normalization template, making it easier for users.

```python
from fuxion.pipelines import DatasetPipeline
from rich import print

output_structure = {
    "title": str,
    "given": str,
    "middle": str,
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
```

The pipeline chain now takes an `output_structure` dictionary that specifies the desired format for the normalized data. This replaces the previous `normalizer_template` parameter, simplifying the process and reducing complexity.

The `output_structure` dictionary defines the fields and their expected types for the normalized output. The pipeline uses this structure to guide the LLM in formatting the generated data correctly.

Other parameters remain similar to the previous version, including `generator_template`, `few_shot_file`, `dataset_name`, number of datapoints to generate `k`, and other configuration options.

The dataset is then saved in a JSON file with the provided dataset name.

<b> Models supported </b>

- gpt-3.5-turbo
- gpt-4
- gpt-4o
- gpt-4o-mini

### Future work:

fuxion is still a work in progress, but it is a good starting point for anyone looking to generate synthetic data for testing and training machine learning models. We plan to add more features to fuxion in the future, including a seamless functionality for accurate data generation and normalization using various llms (locally hosted or via the huggingface api). For now, OpenAI's models are the most functional and reliable.

Feel free to contribute to this project by opening an issue or a pull request. We would love to hear your thoughts on how we can improve fuxion!
