# Datasynth
Langchain + GPT-3* powered data generation and normalization functions. 
# Table of Contents
* [Description](#Description)
* [Installation](#installation)
* [Getting Started](#getting-started)
* [Usage](#usage)
  * [Gneration](#Generation)
  * [Normalization](#Normalization)
  * [Pipelines](#Pipelines)

# Description
Datasynth is a Python package that provides you with a data generation and normalization pipeline which could be used for testing, normalization and training machine learning models. Using this software, you are able to generate sythetic data for different types of use cases -- all that's required is that you pass the write prompt to the chain and watch how things unfold :sunglasses:
# Installation

We recommend that you create a virtual environment before proceeding with the installation process as it would help to create an isolated environment for this project. After doing that, you can proceed with the installation by following the steps below.

- Clone the repository
  ```bash
  git clone [https <repository url>]
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
# Getting started 


# Usage
A couple things to note:
- The project directory contains two folders named `examples` and `templates`. These are folders that contain files for few-shot learning and the prompts that need to be passed to the chain respectively.
- The `templates` directory contains a generator and normalizer directory which hold the generator and normalizer prompts 

By default, we provide three different templates ```[name, address, price]``` which could be used to generate/normalize synthetic data out of the box.

## Generation
Datasynth can be used to generate synthetic data for rapid product testing amongst other use cases. For each generator template, we have a prompt that instructs the chain on what to do. Below is an example of what the `address.template` file looks like

```template
Generate a list of U.S. postal addresses separated by double newlines.  

Make them as realistic and diverse as possible.
Include some company address, P.O. boxes, apartment complexes, etc.
Ensure the addresses are fake.

{{few_shot}}

List:
```

The first line tells the chain to generate addresses

The second block contains a bunch of creative instructions that determines the quality of the results.

`{{few_shot}}` tells the chain to get few-shot examples provided in the examples folder.

`List` returns the results in a list 

With that established, we show how to run the generator script from the terminal 
![](https://github.com/Tobiadefami/datasynth/blob/api-tweaks/terminal_gifs/generator.gif)

## Normalization 
Normalize data into component parts/structured form
![](https://github.com/Tobiadefami/datasynth/blob/api-tweaks/terminal_gifs/normalizers.gif)

## Pipelines
Generating data to train machine learning models

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

from the above description, pipelines requires only the datatype argument to be passed to it per run, while all other arguments are optional.


