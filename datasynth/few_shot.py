import random
from datasynth import EXAMPLE_DIR
import os
import json
from typing import Any

def generate_population(datatype: str) -> list[str]:
    population: list[str] = []
    example_file: str = os.path.join(EXAMPLE_DIR, f"{datatype}.json")
    if os.path.exists(example_file):
        population = json.load(open(example_file))
    return population

def convert_to_string(example: str|dict[str, str]):
    
    if isinstance(example, str):
        return example
    elif isinstance(example, dict):
        new_examples = "\n".join(f"{key}:{value}" for key,value in example.items())
        return new_examples

def populate_few_shot(population:list[Any], sample_size: int):
    sample = random.sample(population, k=sample_size)
    sample_text = "\n---\n".join([convert_to_string(example) for example in sample])
    few_shot = f"Reference Format:\n{sample_text}\n"
    return few_shot