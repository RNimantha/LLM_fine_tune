"""Merge a trained LoRA adapter into the base model to produce a standalone model.

Why merge: shipping the adapter is smallest, but a merged fp16 model is simplest to
serve (no PEFT at load time, works with vLLM/TGI directly).

IMPORTANT: merging must be done in fp16/bf16, NOT in 4-bit. We reload the base in
half precision, attach the adapter, then merge_and_unload().

Run:
    python -m src.merge \
        --adapter outputs/qwen2.5-1.5b-text2sql-qlora \
        --out outputs/qwen2.5-1.5b-text2sql-merged \
        [--push your-username/qwen2.5-1.5b-text2sql]
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(adapter_dir: str, out_dir: str, push_to: str | None) -> None:
    base_id = PeftConfig.from_pretrained(adapter_dir).base_model_name_or_path
    print(f"Base model: {base_id}")

    # Load base in bf16 (full precision relative to the 4-bit training base) so the merge
    # is numerically faithful.
    base = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()
    print("Adapter merged into base weights.")

    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Merged model saved to: {out_dir}")

    if push_to:
        model.push_to_hub(push_to)
        tokenizer.push_to_hub(push_to)
        print(f"Pushed merged model to hub: {push_to}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--push", default=None, help="optional hub repo id")
    args = p.parse_args()
    merge(args.adapter, args.out, args.push)


if __name__ == "__main__":
    main()
