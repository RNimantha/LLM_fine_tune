"""QLoRA supervised fine-tuning for text-to-SQL.

Run:
    python -m src.train --config configs/qlora_qwen.yaml

Design notes (the "why"):
  * 4-bit NF4 quantization freezes the base model and shrinks it ~4x so a 1.5-7B model
    fits on a 16-24 GB Colab GPU.
  * prepare_model_for_kbit_training enables gradient checkpointing hooks, casts layernorms
    to fp32 for stability, and enables input gradients so the (frozen) base can backprop
    into the trainable LoRA adapters.
  * We use the current TRL API: `processing_class=` (not `tokenizer=`) and
    `SFTConfig(max_length=...)` (not `max_seq_length=`). Older tutorials will crash here.
"""

from __future__ import annotations

import argparse

import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from src.config import Config, load_config
from src.data import load_and_format, preview

_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def build_tokenizer(cfg: Config) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(
        cfg.model.base_model, trust_remote_code=cfg.model.trust_remote_code
    )
    # Causal LMs must pad LEFT for correct generation; SFT itself is robust but we set a
    # pad token explicitly to avoid the eos==pad ambiguity that silently corrupts loss masks.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # right padding is fine/standard for training
    return tok


def build_model(cfg: Config):
    """Load the base model in 4-bit and prep it for k-bit training."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg.quant.load_in_4bit,
        bnb_4bit_quant_type=cfg.quant.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.quant.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=_DTYPE[cfg.quant.bnb_4bit_compute_dtype],
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=cfg.model.trust_remote_code,
        attn_implementation="eager",  # most compatible on T4; use "flash_attention_2" on A100
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.training.gradient_checkpointing
    )
    return model


def build_lora_config(cfg: Config) -> LoraConfig:
    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        target_modules=cfg.lora.target_modules,
        task_type="CAUSAL_LM",
    )


def build_sft_config(cfg: Config) -> SFTConfig:
    t = cfg.training
    return SFTConfig(
        output_dir=t.output_dir,
        num_train_epochs=t.num_train_epochs,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        warmup_ratio=t.warmup_ratio,
        weight_decay=t.weight_decay,
        max_length=t.max_length,            # NOTE: renamed from max_seq_length
        packing=t.packing,
        gradient_checkpointing=t.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=t.optim,
        bf16=t.bf16,
        logging_steps=t.logging_steps,
        eval_strategy=t.eval_strategy,
        eval_steps=t.eval_steps,
        save_strategy=t.save_strategy,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        report_to=t.report_to,
        seed=t.seed,
        dataset_num_proc=2,
    )


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    tokenizer = build_tokenizer(cfg)
    ds = load_and_format(
        dataset_name=cfg.data.dataset_name,
        test_size=cfg.data.test_size,
        seed=cfg.data.seed,
        max_train_samples=cfg.data.max_train_samples,
        max_eval_samples=cfg.data.max_eval_samples,
    )
    print(f"Train: {len(ds['train'])} | Test: {len(ds['test'])}")
    preview(ds, tokenizer, n=1)  # sanity-check formatting before burning GPU time

    model = build_model(cfg)

    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(cfg),
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        peft_config=build_lora_config(cfg),
        processing_class=tokenizer,   # NOTE: replaces deprecated `tokenizer=`
    )

    # Confirm we're training <1% of params — the whole point of QLoRA.
    trainer.model.print_trainable_parameters()

    trainer.train()

    # Save the adapter (small) + tokenizer. Use src/merge.py to produce a standalone model.
    trainer.save_model(cfg.training.output_dir)
    tokenizer.save_pretrained(cfg.training.output_dir)
    print(f"\nAdapter saved to: {cfg.training.output_dir}")

    if cfg.hub.push_to_hub and cfg.hub.hub_model_id:
        trainer.push_to_hub(cfg.hub.hub_model_id)
        print(f"Pushed adapter to hub: {cfg.hub.hub_model_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/qlora_qwen.yaml")
    args = p.parse_args()
    main(args.config)
