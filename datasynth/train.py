from transformers import TrainingArguments
import fire
import typer
from transformers import Trainer, AutoConfig
from datasynth.dataset import tokenizer, NormalizationDataset
from transformers import DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration
import wandb


def test_generation(model, test_example):
    example = test_example["input_ids"].unsqueeze(0).to(model.device)
    output_tokens = model.generate(input_ids=example, max_length=256, temperature=0.0)
    output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    expected_output_text = tokenizer.decode(
        test_example["labels"], skip_special_tokens=True
    )
    print("Output", output_text)
    print("Expected Output", expected_output_text)


def main(
    experiment_name,
    data_file: str | None = None,
    dataset_size: int | None = None,
    dataloader_num_workers: int = 0,
    max_length: int = 512,
    batch_size: int = 2,
    dropout: float = 0.0,
    gradient_checkpointing: bool = False,
    pretrained_checkpoint: str | None = None,
    base_model: str = "google/byt5-small",
    num_train_epochs: float = 10.0,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    gradient_accumulation_steps: int = 1,
    # resume: bool=False,
    # max_steps: int = None,
):
    # TODO: start training from random initialization
    # TODO: incorporate other objectives
    model_cls = T5ForConditionalGeneration

    if pretrained_checkpoint is None:
        print("Training from pre-trained model")
        pretrained_checkpoint = base_model
        print(pretrained_checkpoint)

    config = AutoConfig.from_pretrained(pretrained_checkpoint)
    config.hidden_dropout_prob = dropout
    config.attention_probs_dropout_prob = dropout
    model = model_cls.from_pretrained(pretrained_checkpoint, config=config)

    args = TrainingArguments(
        output_dir=experiment_name,
        run_name=experiment_name,
        dataloader_num_workers=dataloader_num_workers,
        per_device_train_batch_size=batch_size,
        do_eval=False,
        evaluation_strategy="no",
        num_train_epochs=num_train_epochs,
        prediction_loss_only=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        ignore_data_skip=False,
        save_steps=1000,
        save_total_limit=2,
        save_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        # max_steps=max_steps,
        report_to="wandb",
        gradient_checkpointing=gradient_checkpointing
        # fp16=True,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        max_length=max_length,
        return_tensors="pt",
        padding="max_length",
    )

    train_dataset = NormalizationDataset(
        json_file=data_file, max_length=max_length, dataset_size=dataset_size
    )
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    test_generation(model, test_example=train_dataset[0])
    trainer.save_model()


if __name__ == "__main__":
    typer.run(main)
