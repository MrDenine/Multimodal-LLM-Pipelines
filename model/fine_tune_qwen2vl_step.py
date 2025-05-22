from typing import Any, Annotated

from accelerate import PartialState
from qwen_vl_utils import process_vision_info
from sympy.printing.pytorch import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from trl import SFTConfig, SFTTrainer
from zenml import step, log_metadata

from steps.utils import print_gpu_utilization


@step
def fine_tune_qwen2vl(
    model_id: str,
    train_set: Any,
    eval_set: Any,
    output_dir: str = "qwen2vl-model-2b-instruct-spatial-information",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    logging_steps: int = 5,
) -> tuple[Qwen2VLForConditionalGeneration, Any]:

    log_metadata(
    metadata={
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "logging_steps": logging_steps,
        }
    )

    device_string = PartialState().process_index

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )

    processor = Qwen2VLProcessor.from_pretrained(model_id)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=learning_rate,
        lr_scheduler_type="constant",
        logging_steps=logging_steps,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to=["none"],
    )
    training_args.remove_unused_columns = False

    def collate_fn(examples):
        texts = [
            processor.apply_chat_template(example, tokenize=False) for example in examples
        ]
        image_inputs = [process_vision_info(example)[0] for example in examples]

        batch = processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        image_tokens = [151652, 151653, 151655]
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100

        batch["labels"] = labels
        return batch

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,
    )
    trainer.train()

    return model, processor