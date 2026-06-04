# Fine-Tuning LLMs with Hugging Face — A-to-Z Study Plan (Colab Pro)

A 6-week, project-driven curriculum. You don't study theory then build — you build the
`text-to-SQL` project in this repo *while* learning each concept. Every week ends with a
commit to GitHub so your learning is visible and your portfolio grows.

> Target outcome: you can take any base LLM, fine-tune it with QLoRA on a domain task,
> evaluate it with a real metric, merge + publish it to the Hub, and explain every design
> decision in an interview.

---

## Mental model first (read once, refer back often)

There are three "levers" in modern fine-tuning. Almost every decision maps to one:

1. **What you adapt** — Full fine-tune vs **PEFT** (LoRA / **QLoRA**). On Colab you will
   *always* use QLoRA: the base model is frozen and 4-bit quantized, and you train a tiny
   set of low-rank adapter matrices (<1% of params). This is what makes a 7B model trainable
   on a 16 GB T4.
2. **What you teach** — the *objective*. **SFT** (supervised fine-tuning on input→output
   pairs) is 90% of industry work. Preference tuning (**DPO/ORPO**) comes after, only if SFT
   isn't enough. You'll do SFT here.
3. **How you measure** — without a task metric you are flying blind. Loss going down ≠ model
   getting useful. This project uses **SQL AST-match + exact-match accuracy** on a held-out set.

Keep these three in your head. When something breaks or a choice feels arbitrary, ask:
"which lever is this?"

---

## Week 0 — Environment & foundations (½ day)

**Goal:** Colab Pro set up, repo cloned, you can load a model and generate text.

- Colab Pro: `Runtime → Change runtime type → GPU`. You'll get **T4 (16 GB)** most of the
  time; sometimes **L4 (24 GB)** or **A100 (40 GB)** with "High-RAM"/"Premium GPU". Check with
  `!nvidia-smi`. The whole repo is tuned to run on a plain T4.
- Understand the 4 libraries you'll live in:
  - `transformers` — models + tokenizers + `Trainer`.
  - `peft` — LoRA/QLoRA adapters.
  - `trl` — `SFTTrainer`/`SFTConfig`, the high-level SFT loop.
  - `bitsandbytes` — 4-bit quantization (the "Q" in QLoRA).
  - `datasets` — streaming/loading data; `accelerate` — device placement.
- **Do:** open `notebooks/colab_finetune_text2sql.ipynb`, run cells 1–2 (install + smoke test).
- **Commit:** `chore: env setup + smoke test`.

**Interview hooks:** What is the difference between `transformers.Trainer` and `trl.SFTTrainer`?
(SFTTrainer wraps Trainer; it handles chat-template formatting, packing, and PEFT wiring.)

---

## Week 1 — Tokenization, chat templates, and data formatting

This is where most fine-tunes silently fail. Get it right and the rest is easy.

- **Tokenizers:** subword (BPE/Unigram), `input_ids`, `attention_mask`, special tokens,
  `pad_token` vs `eos_token`, left vs right padding (causal LMs pad **left** for generation).
- **Chat templates:** every instruct model ships a Jinja `chat_template`. You must format your
  data with the *same* template the model was trained on, or you get garbage. Learn
  `tokenizer.apply_chat_template(messages, tokenize=False)`.
- **The two dataset formats `SFTTrainer` accepts:**
  - *Conversational*: a `messages` column = list of `{"role","content"}`. SFTTrainer applies
    the chat template for you. **Use this.** (`src/data.py`)
  - *Standard*: a `text` column with the already-formatted string.
- **Completion-only loss:** you usually want loss on the *assistant* tokens only, not the
  prompt. Know that `assistant_only_loss`/`completion_only_loss` exist and when to use them.
- **Do:** study `src/data.py`. Run the data-inspection cell — print 2 formatted examples and
  confirm the SQL appears after the assistant turn.
- **Commit:** `feat: data loading + chat formatting for sql-create-context`.

**Interview hooks:** Why must train-time formatting match the model's chat template?
What breaks if you pad right for a decoder-only model at inference?

---

## Week 2 — PEFT, LoRA, and QLoRA (the core)

- **Why not full fine-tune:** a 7B model in fp16 needs ~14 GB just for weights, plus
  gradients + optimizer states (~4×) → ~80–100 GB. Impossible on Colab.
- **LoRA:** freeze `W`, learn `W + B·A` where `A`,`B` are low-rank (`r`). You train `r·(d_in+d_out)`
  params per layer instead of `d_in·d_out`. `lora_alpha` scales the update (`alpha/r`).
- **QLoRA = LoRA on a 4-bit (NF4) quantized base.** Base weights are frozen and stored in 4-bit;
  adapters train in bf16. This is the single most important technique for you.
- **Key knobs (in `configs/qlora_qwen.yaml`):**
  - `r` (8–64; start 16), `lora_alpha` (usually 2×`r`), `lora_dropout` (0.05).
  - `target_modules`: attention + MLP projections (`q,k,v,o,gate,up,down`). All-linear is the
    QLoRA-paper recommendation and what we use.
  - `bnb_4bit_quant_type="nf4"`, `bnb_4bit_use_double_quant=True`,
    `bnb_4bit_compute_dtype=bfloat16`.
- **Do:** read `src/train.py`, especially `build_model()` and `build_lora_config()`. Print
  `model.print_trainable_parameters()` — confirm you're training <1%.
- **Commit:** `feat: QLoRA training script with bitsandbytes 4-bit + LoRA`.

**Interview hooks:** Explain QLoRA in two sentences. What does `lora_alpha` do? Why NF4 over
plain int4? What does `prepare_model_for_kbit_training` change (gradient checkpointing,
input-grad enabling, layernorm in fp32)?

---

## Week 3 — Run the training loop, read the curves

- **`SFTConfig` essentials:** `per_device_train_batch_size`, `gradient_accumulation_steps`
  (effective batch = product × #GPUs), `learning_rate` (2e-4 is a good QLoRA default),
  `lr_scheduler_type="cosine"`, `warmup_ratio`, `num_train_epochs` (1–3), `bf16=True`,
  `gradient_checkpointing=True`, `optim="paged_adamw_8bit"`, `max_length`, `packing`.
- **VRAM survival kit (when you OOM):** lower `max_length` → lower batch size → raise
  `gradient_accumulation_steps` → enable `gradient_checkpointing` → `packing=True` →
  `optim="paged_adamw_8bit"` → smaller base model.
- **Reading loss:** train loss should fall smoothly; if eval loss rises while train falls →
  overfitting (fewer epochs / more dropout / more data). Flat loss → LR too low or data
  malformed.
- **Experiment tracking:** wire up `report_to="tensorboard"` (or W&B). You should never tune
  blind.
- **Do:** run the full training cell on a small subset first (`max_train_samples=2000`,
  1 epoch) end-to-end, then scale up.
- **Commit:** `feat: full SFT run + tensorboard logging`.

**Interview hooks:** What's the effective batch size with bs=2, grad_accum=8, 1 GPU? Why
gradient accumulation? Cosine vs linear schedule trade-off?

---

## Week 4 — Evaluation that actually means something

- **The trap:** reporting only training loss. Recruiters and senior reviewers ignore it.
- **This project's metric (`src/evaluate.py`):**
  - **Exact-match** after normalization (lowercasing, whitespace, alias canonicalization).
  - **AST-match** via `sqlglot`: parse gold + predicted SQL into ASTs and compare — robust to
    formatting/ordering differences. This is the credible number.
  - **Parse-rate:** % of generated SQL that is even valid SQL.
- **Build a held-out test split** the model never saw. Report all three numbers + a few
  qualitative failure cases.
- **Do:** run evaluation on the base model *and* your fine-tuned model. The delta is your
  headline result ("base 38% → fine-tuned 81% AST-match").
- **Commit:** `feat: SQL AST/exact-match evaluation + base-vs-finetuned comparison`.

**Interview hooks:** Why is exact-match a bad SQL metric on its own? What is execution
accuracy and why is it the gold standard (and why is it hard to set up)?

---

## Week 5 — Merge, publish, serve, and write it up

- **Adapters vs merged model:** you can ship the tiny LoRA adapter (load base + adapter at
  runtime) *or* `merge_and_unload()` into a standalone model. Know both; `src/merge.py` does
  the merge + optional Hub push.
- **Push to Hugging Face Hub:** model card with the eval table, intended use, limitations.
  This *is* a portfolio piece.
- **Inference:** `src/infer.py` — batched generation, correct generation config
  (`do_sample=False` for SQL, `temperature` only if sampling), stop tokens.
- **Optional serving:** quick `transformers` pipeline; mention vLLM/TGI for real throughput.
- **README + model card:** problem → data → method → results table → how to reproduce. This
  is what a hiring manager reads first.
- **Commit/tag:** `release: v1.0 fine-tuned text2sql model + model card`, push, tag `v1.0`.

**Interview hooks:** When do you ship the adapter vs the merged model? What goes in a model card?

---

## Week 6 — Go beyond (pick one, shows seniority)

- **DPO/ORPO:** take your SFT model and preference-tune on chosen/rejected SQL pairs (`trl`
  has `DPOTrainer`/`ORPOTrainer`). Great "what's next" answer.
- **Execution-accuracy eval:** build SQLite DBs from the `CREATE TABLE` context, run gold vs
  predicted, compare result sets. The gold-standard metric.
- **Function-calling / tool-use fine-tune:** reuse the whole pipeline on a tool-calling
  dataset — directly relevant to agentic work.
- **Quantize for deployment:** GGUF (llama.cpp) or AWQ/GPTQ for a small, fast artifact.
- **Data quality loop:** dedupe, filter, and length-balance the dataset; re-measure. Often
  beats any hyperparameter tweak.

---

## What "done" looks like (portfolio checklist)

- [ ] Public GitHub repo with clean README + results table.
- [ ] Fine-tuned model (or adapter) on the HF Hub with a real model card.
- [ ] A base-vs-finetuned eval comparison with numbers.
- [ ] You can whiteboard QLoRA, the SFT data path, and your eval metric without notes.

---

## Reference: the minimal correct SFT loop (memorize the shape)

```python
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

trainer = SFTTrainer(
    model=model,                       # 4-bit base, kbit-prepared
    train_dataset=ds["train"],         # has a "messages" column
    eval_dataset=ds["test"],
    peft_config=LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM"),
    processing_class=tokenizer,        # NOTE: not `tokenizer=` (removed in trl>=0.16)
    args=SFTConfig(
        output_dir="out",
        max_length=1024,               # NOTE: not `max_seq_length`
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    ),
)
trainer.train()
```

If you can explain every line above, you understand fine-tuning.
