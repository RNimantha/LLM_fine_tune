# Fine-Tuning an LLM for Text-to-SQL with QLoRA

Fine-tune a small instruction-tuned LLM (Qwen2.5) to translate natural-language questions
into SQL, using **QLoRA** on a single Colab Pro GPU. End-to-end and production-shaped:
typed configs, a real evaluation metric (SQL **AST-match**), adapter merging, and
Hub publishing.

> **Why text-to-SQL?** It's a real, in-demand task (analytics copilots, BI assistants),
> it has a clean public dataset, and—unlike vague "chatbot" fine-tunes—it has an
> **objective, measurable metric**. That measurable delta (base vs fine-tuned) is what
> makes this a portfolio piece, not a toy.

---

## Results

Fill these in after your run (`python -m src.evaluate ...` for each model). Example shape:

| Model                              | parse_rate | exact_match | **ast_match** |
| ---------------------------------- | ---------- | ----------- | ------------- |
| Qwen2.5-1.5B-Instruct (base)       |  ~0.90     |  ~0.20      |  **~0.38**    |
| **Qwen2.5-1.5B + QLoRA (ours)**    |  ~0.99     |  ~0.55      |  **~0.81**    |

`ast_match` is the headline: queries that are *structurally equivalent* to the gold SQL,
robust to formatting/aliasing. (Numbers above are illustrative—report your own.)

---

## Quickstart (Colab Pro — recommended)

1. Open `notebooks/colab_finetune_text2sql.ipynb` in Colab.
2. `Runtime → Change runtime type → GPU` (T4 is enough; L4/A100 lets you use a bigger model).
3. Run all cells top to bottom: install → inspect data → train → evaluate → infer → (merge/push).

## Quickstart (local / any machine with a CUDA GPU)

```bash
git clone https://github.com/<you>/llm-finetuning-text2sql.git
cd llm-finetuning-text2sql
pip install -r requirements.txt

# 1. Train (QLoRA SFT)
python -m src.train --config configs/qlora_qwen.yaml

# 2. Evaluate base vs fine-tuned
python -m src.evaluate --model Qwen/Qwen2.5-1.5B-Instruct --limit 300        # baseline
python -m src.evaluate --model outputs/qwen2.5-1.5b-text2sql-qlora --limit 300 # tuned

# 3. Try it
python -m src.infer --model outputs/qwen2.5-1.5b-text2sql-qlora \
  --schema "CREATE TABLE head (age INTEGER)" \
  --question "How many heads are older than 56?"

# 4. (optional) Merge adapter -> standalone model, push to Hub
python -m src.merge --adapter outputs/qwen2.5-1.5b-text2sql-qlora \
  --out outputs/merged --push <you>/qwen2.5-1.5b-text2sql
```

---

## Project structure

```
llm-finetuning-text2sql/
├── README.md                       # you are here
├── CURRICULUM.md                   # the A-to-Z study plan (read this first)
├── requirements.txt
├── configs/
│   └── qlora_qwen.yaml             # all hyperparameters, one place
├── src/
│   ├── config.py                   # typed (Pydantic) config loader
│   ├── data.py                     # dataset load + chat-template formatting
│   ├── train.py                    # QLoRA SFT (current TRL API)
│   ├── evaluate.py                 # AST-match / exact-match / parse-rate
│   ├── merge.py                    # merge LoRA into base + Hub push
│   └── infer.py                    # generation (adapter or merged model)
├── notebooks/
│   └── colab_finetune_text2sql.ipynb
└── scripts/
    └── run.sh                      # convenience wrapper
```

---

## Method (one paragraph for your interview)

The base model is loaded in **4-bit NF4** (frozen) and we train **LoRA adapters** on all
linear projections (`q,k,v,o,gate,up,down`, `r=16`, `alpha=32`) — under 1% of parameters.
Data is the `b-mc2/sql-create-context` dataset, formatted into the model's **chat template**
so train-time formatting matches the instruct model's pretraining. We supervise with
`trl`'s `SFTTrainer` (`paged_adamw_8bit`, cosine schedule, `lr=2e-4`, gradient
checkpointing). We evaluate with **`sqlglot` AST equivalence** on a held-out split, compare
against the un-tuned base, then merge the adapter and publish a model card.

## Hardware notes

| GPU (Colab tier)        | VRAM   | Suggested base model            | batch / grad_accum |
| ----------------------- | ------ | ------------------------------- | ------------------ |
| T4 (Pro, common)        | 16 GB  | Qwen2.5-1.5B-Instruct           | 2 / 8              |
| L4 (Pro, sometimes)     | 24 GB  | Qwen2.5-3B-Instruct             | 4 / 4              |
| A100 (Pro+, premium)    | 40 GB  | Qwen2.5-7B-Instruct             | 8 / 2              |

**If you OOM:** lower `max_length` → lower batch size → raise `gradient_accumulation_steps`
→ keep `gradient_checkpointing: true` → `packing: true` → smaller model.

## Common gotchas (these break most online tutorials)

- `trl>=0.16` removed `tokenizer=` → use **`processing_class=`**.
- `SFTConfig` uses **`max_length`**, not `max_seq_length`.
- `dataset_kwargs=` was removed from `SFTTrainer.__init__`.
- Set `model.config.use_cache = False` when using gradient checkpointing.
- Merge adapters in **bf16/fp16**, never while the base is still 4-bit.

## License

MIT. Dataset and base model carry their own licenses—check before commercial use.
