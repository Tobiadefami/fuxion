import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Any

tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

def preprocess(example: dict[str, Any], max_length: int = 512) -> dict[str, Any]:

    encoded_inputs = tokenizer(
        example["generated_input"],
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        truncation=True,
        return_overflowing_tokens=False,
    )

    encoded_outputs = tokenizer(
        json.dumps(example["normalized_output"]),
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        truncation=True,
        return_overflowing_tokens=False,
    )
    result = {
        "input_ids": encoded_inputs.input_ids[0],
        "attention_mask": encoded_inputs.attention_mask[0],
        "labels": encoded_outputs.input_ids[0],
    }
    return result


class NormalizationDataset(Dataset):
    def __init__(
        self,
        json_file,
        max_length=2048,
        dataset_size=None,
    ):

        with open(json_file) as fd:
            self.dataset = json.load(fd)["dataset"]["outputs"]

        if dataset_size is not None:
            self.dataset = self.dataset[:dataset_size]
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        example = self.dataset[index]
        processed_examples = preprocess(example, self.max_length)
        return processed_examples


if __name__ == "__main__":
    dataset = NormalizationDataset(
        json_file="/home/chief/datasynth/datasynth/datasets/qa1000.json",
    )
    print(dataset[15])
