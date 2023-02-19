from dataset import DocModelDataset
from transformers import TrainingArguments
from collator import DataCollatorForWholeWordMask
import fire
import torch
from docmodel.doc_model import RobertaDocModelForMLM, XDocModelForMLM, DocModelConfig
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizerFast, RobertaConfig
from transformers import Trainer, AutoConfig
from datasynth.dataset import tokenizer, NormalizationDataset
from transformers import DataCollatorForSeq2Seq



def main(
    experiment_name,
    data_file=None,
    dataloader_num_workers=0,
    max_length=512,
    batch_size=2,
    dropout=0.,
    gradient_checkpointing=True,
    pretrained_checkpoint=None,
    base_model="google/byt5-small",
    num_train_epochs=1.0,
    learning_rate=3e-4,  
    weight_decay=0.01,  
    warmup_ratio=0.1,
    gradient_accumulation_steps=8,
    resume=False,
    max_steps=10000,
    **kwargs
):
    if kwargs:
        raise AssertionError(f"Unexpected arguments: {kwargs}")
    # TODO: start training from random initialization
    # TODO: incorporate other objectives
    model_cls = T5ForConditionalGeneration

    print("Training from pre-trained model")
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
        max_steps=max_steps,
        report_to="wandb",
        # fp16=True,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, max_length=max_length, return_tensors="pt"
    )
    

    train_dataset = NormalizationDataset(
        json_file=data_file,
        max_length=max_length
    )
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    fire.Fire(main)
