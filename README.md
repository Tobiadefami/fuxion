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

## Generation

fuxion can be used to generate synthetic data for rapid product testing amongst other use cases. For each generator template, we have a prompt that instructs the chain on what to do. Below is an example of what the `address.template` file looks like

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



## Normalization 

It's often necessary to transform data into a standardized form before storing in a database. Using fuxion, you can make unstructured data useful by breaking it up into it's component parts and normalizing into a more structured form. Like the generator example, we have a normalization template that contains prompts that instructs the chain on how to achieve this. 

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
We can train machine learning models on the combination of synthetically generated data and their normalized format. This is where we use `pipelines.py` 
