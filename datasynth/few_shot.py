import random
from datasynth import EXAMPLE_DIR
import os
import json


def generate_population(datatype: str):
    population: list[str] = []
    example_file: str = os.path.join(EXAMPLE_DIR, f"{datatype}.json")
    if os.path.exists(example_file):
        population = json.load(open(example_file))
    return population


def populate_few_shot(population:list[str], sample_size: int):
    sample = random.sample(population, k=sample_size)
    sample_text = "\n\n".join(sample)
    few_shot = f"Reference Format:\n{sample_text}\n"
    return few_shot