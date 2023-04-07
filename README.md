# Datasynth
LangChain + LLM powered data generation and normalization functions. 
Datasynth helps you generate a fully synthetic dataset with LLM APIs to train a task-specific model you can run on your own GPU.
Preliminary models for name, price, and address standardization are available on [HuggingFace](https://huggingface.co/PragmaticMachineLearning).

![datasynth](/assets/datasynth.png)
# Table of Contents
* [Description](#description)
* [Installation](#installation)
* [Usage](#usage)
  * [Generation](#generation)
  * [Normalization](#normalization)
  * [Pipelines](#pipelines)

# Description
Datasynth is a Python package that provides you with a data generation and normalization pipeline which could be used for testing, normalization and training machine learning models. Using this software, you are able to generate sythetic data for different types of use cases -- all that's required is that you pass the right prompt to the chain and watch how things unfold :sunglasses:

# Installation

We recommend that you create a virtual environment before proceeding with the installation process as it would help to create an isolated environment for this project. After doing that, you can proceed with the installation by following the steps below.

- Clone the repository
  ```bash
  git clone git@github.com:Tobiadefami/datasynth.git
  ```
- Install poetry: 
  ```bash
  pip install poetry  
  ```
- From the terminal, cd into the project directory and run: 
  ```bash 
  poetry install
  ```
- Add the following to your bashrc file and replace "your-key" with your OpenAI API key: 
  ```bash
  export OPENAI_API_KEY = "your-key"
  ```   

# Usage

A couple things to note:
- The project directory contains two folders named `examples` and `templates`. These are folders that contain files for few-shot learning and the prompts that need to be passed to the chain respectively.
- The `templates` directory contains a generator and normalizer directory which hold the generator and normalizer prompts 

By default, we provide three different templates ```[name, address, price]``` which could be used to generate/normalize synthetic data out of the box.

## Generation

Datasynth can be used to generate synthetic data for rapid product testing amongst other use cases. For each generator template, we have a prompt that instructs the chain on what to do. Below is an example of what the `address.template` file looks like

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

With that established, we show how to run the generator script from the terminal 

![](https://github.com/Tobiadefami/datasynth/blob/api-tweaks/terminal_gifs/generator.gif)

## Normalization 

It's often necessary to transform data into a standardized form before storing in a database. Using `datasynth`, can make unstructured data useful by breaking it up into it's component parts and normalizing into a more structured form. Like the generator example, we have a normalization template that contains prompts that instructs the chain on how to achieve this. 

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

You can run the normalizer by passing it the datatype (name of data to be generated), and an "example"

![](https://github.com/Tobiadefami/datasynth/blob/api-tweaks/terminal_gifs/normalizer.gif)

## Pipelines
We can train machine learning models on the combination of synthetically generated data and their normalized format. This is where we use `pipelines.py` 

```bash
Usage: pipelines.py [OPTIONS] DATATYPE                                                                                                               
╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    datatype      TEXT  [default: None] [required]                                                                                                                                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --k                             INTEGER  [default: 10]                                                                                                                                                                                                   │
│ --dataset-name                  TEXT     [default: None]                                                                                                                                                                                                 │
│ --temperature                   FLOAT    [default: 0.8]                                                                                                                                                                                                  │
│ --cache           --no-cache             [default: no-cache]                                                                                                                                                                                             │
│ --help                                   Show this message and exit.                                                                                                                                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

From the output above, the only required argument to be passed to `pipelines.py` is the `datatype`, which is just the name of the type of data you want to generate. Other optional arguments like `k` determines the number of samples to be generated and is set to `10` by default. 

Running:
```bash
python pipelines.py address --k 20 --dataset-name sample-address
```
![](https://github.com/Tobiadefami/datasynth/blob/api-tweaks/terminal_gifs/pipeline.gif)
Generates 20 samples of generated addresses with their normalized outputs strored in `sample-address.json`. Here is a [link](https://github.com/Tobiadefami/datasynth/blob/api-tweaks/datasynth/datasets/sample-address.json) to the file



