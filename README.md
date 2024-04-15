# fuxion
LangChain + LLM powered data generation and normalization functions. 
fuxion helps you generate a fully synthetic dataset with LLM APIs to train a task-specific model you can run on your own GPU.
Preliminary models for name, price, and address standardization are available on [HuggingFace](https://huggingface.co/PragmaticMachineLearning).

![fuxion](/assets/fuxion.png)
# Table of Contents
* [Description](#description)
* [Installation](#installation)
* [Usage](#usage)
  * [Generation](#generation)
  * [Normalization](#normalization)
  * [Template Structure](#template-structure)
  * [Pipelines](#pipelines)

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
  export OPENAI_API_KEY = "your-key"
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

* The first few lines tells the chain to generate addresses and contains a bunch of creative instructions that determines the quality of the results.

* `{{few_shot}}` tells the chain to get few-shot examples provided in the examples folder.

* `List` returns the results in a list 

> The same convention should be followed when creating subsequent templates for various data generation tasks.


##### Normalizer templates

``` 

Format the following address as a list of python dictionaries of the form:
[
    { 
        "house_number": int, 
        "road": str, 
        "unit": int, 
        "unit_type": str, 
        "po_box_number": int, 
        "city": str, 
        "state": str, 
        "postcode": int 
    }
]. 

Use abbreviations for state and road type.
Use short form zip codes.

Input:
"{{address}}"

Output:
[{
```

* The first few lines tells the chain to format the address passed to it into a list of dict(s)

* It then takes in `{{address}}` as input
* And returns a list of dict as output



## Pipelines
We can train machine learning models on the combination of synthetically generated data and their normalized format. This is where we use `pipelines` 

```python

from fuxion.pipelines import DatasetPipeline

pipeline_chain = DatasetPipeline.from_template(
    generator_template="examples/name_generator/generator.template",
    normalizer_template="examples/name_generator/normalizer.template",
    few_shot_file="examples/name_generator/few_shot.json",
    dataset_name="name_pipeline",
    k=20,
    model_name="gpt-3.5-turbo",
    cache=False,
    verbose=True,
    temperature=1.0,
    batch_save=True,
    batch_size = 3,
)

result = pipeline_chain.execute()
print(result)
```

The pipeline chain takes in the `generator_template`, `normalizer_template`, `few shot_file` for few shot examples, `dataset_name`, number of datapoints to generate `k` and other parameters to generate a dataset including the `model_name` argument which specifies the llm to use for the generation and normalization process. The dataset is then saved in a json file with the dataset name provided. The user can choose to save the dataset in batches by setting `batch_save` to `True` and providing a `batch_size`.


<b> Models supported </b>

* gpt-3.5-turbo 
* gpt-4
* gpt-4-1106-preview 
* gpt-3.5-turbo-instruct


### Future work: 

fuxion is still a work in progress, but it is a good starting point for anyone looking to generate synthetic data for testing and training machine learning models. We plan to add more features to fuxion in the future, including a seamless functionality for accurate data generation and normalization using various llms (locally hosted or via the huggingface api). For now, OpenAI's models are the most functional and reliable. 

Feel free to contribute to this project by opening an issue or a pull request. We would love to hear your thoughts on how we can improve fuxion!
