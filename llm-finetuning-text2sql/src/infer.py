"""Inference for the fine-tuned text-to-SQL model.

Works with either:
  * a LoRA adapter dir (base loaded in 4-bit + adapter applied), or
  * a merged standalone model dir (see src/merge.py).

Run (single question):
    python -m src.infer --model outputs/qwen2.5-1.5b-text2sql-qlora \
        --schema "CREATE TABLE head (age INTEGER)" \
        --question "How many heads are older than 56?"

For SQL we use GREEDY decoding (do_sample=False): we want the single most likely,
deterministic query, not creative variation.
"""

from __future__ import annotations

import argparse
import os

import torch
from peft import PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data import SYSTEM_PROMPT, build_user_prompt


def _is_adapter_dir(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left padding for batched generation

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if _is_adapter_dir(model_path):
        from peft import PeftModel

        base_id = PeftConfig.from_pretrained(model_path).base_model_name_or_path
        base = AutoModelForCausalLM.from_pretrained(
            base_id, quantization_config=bnb, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb, device_map="auto"
        )

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_sql(model, tokenizer, schema: str, question: str, max_new_tokens: int = 256) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(schema, question)},
    ]
    # add_generation_prompt=True appends the assistant-turn marker so the model continues
    # as the assistant rather than predicting another user turn.
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,                       # deterministic — correct for SQL
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Decode only the newly generated tokens (strip the prompt).
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="adapter dir or merged model dir")
    p.add_argument("--schema", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)
    sql = generate_sql(model, tokenizer, args.schema, args.question, args.max_new_tokens)
    print(sql)


if __name__ == "__main__":
    main()
