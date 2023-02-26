from transformers import TrainingArguments
import fire
import typer
from transformers import Trainer, AutoConfig
from datasynth.dataset import tokenizer, NormalizationDataset
from transformers import DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration
import wandb


def main(
    checkpoint: str,
    test_data_file: str,
    max_length: int = 512,
    batch_size: int = 4,
    dataset_size: int | None = None,
):
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        max_length=max_length,
        return_tensors="pt",
        padding="max_length",
    )

    test_dataset = NormalizationDataset(
        json_file=test_data_file, max_length=max_length, dataset_size=dataset_size
    )

    results = {"correct": 0, "incorrect": 0}
    for batch_start in range(0, len(test_dataset), batch_size):
        # Making a batch to send to the model
        batch_end = min(batch_start + batch_size, len(test_dataset))
        examples = []
        for sample_idx in range(batch_start, batch_end):
            examples.append(test_dataset[sample_idx])

        # Using the collator to make sure all examples in our batch have the same shape
        # Padding and converting to torch tensors
        batch = collator(examples)

        # Running the model to produce a prediction for normalization
        output = model.generate(
            input_ids=batch["input_ids"], max_length=256, temperature=0.0
        )
        output_texts = tokenizer.batch_decode(output, skip_special_tokens=True)
        expected_texts = tokenizer.batch_decode(
            batch["labels"], skip_special_tokens=True
        )
        input_texts = tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )
        for input_text, output_text, expected_text in zip(
            input_texts, output_texts, expected_texts
        ):
            if output_text.strip() == expected_text.strip():
                status = "correct"
            else:
                status = "incorrect"

            results[status] += 1
            print(
                f"Input: {input_text}\nOutput: {output_text}\nExpected: {expected_text}\nStatus: {status}\n-------"
            )
            acc = results["correct"] / (results["correct"] + results["incorrect"])
            print(
                f"\n\nCurrent Acc: {acc:0.2f} or {results['correct']}/{results['correct'] + results['incorrect']}\n\n"
            )


if __name__ == "__main__":
    typer.run(main)
