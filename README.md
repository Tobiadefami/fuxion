# Datasynth
GPT-3 powered data generation and normalization functions.
# Table of Contents

* [Installation](#installation)
* [Getting Started](#getting-started)
* [Usage](#usage)
  * [Gneration](#Generation)
  * [Normalization](#Normalization)
  * [Pipelines](#Pipelines)

# Installation

We recommend that you create a virtual environment before proceeding with the installation process as it would help to create an isolated environment for this project. After doing that, you can proceed with the installation by following the steps below.

- Clone the repository
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

All of the magic happens in the pipelines script, but inorder to run it some things need to be considered:

- The project directory contains two folders named `examples` and `templates`. These are folders that contain files for few-shot learning and the prompts that need to be passed to the chain respectively.
- The `templates` directory contains a generator and normalizer directory which hold the generator and normalizer prompts 

By default, we provide three different templates ```[name, address, price]``` which generates synthetic data for 

## Generation
Generating synthetic data for testing
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
│ --help                                   Show this message and exit. 
```

from the above description, pipelines requires only the datatype argument to be passed to it per run, while all other arguments are optional.



### Couple things to check: 

1. Add basic tests [done]
2. Verify that all pipelines work [done]
3. Write a README to help with the installation and usage of the project [wip]
4. Generate datapoints for training 
5. We should train basic checkpoints for all of our current pipelines
6. Upload checkpoints to huggingface [not a neccessity]
7. If API timeout(s) we could write files to disk or implement retry logic [wip]


