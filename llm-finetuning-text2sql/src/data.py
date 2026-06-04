"""Dataset loading + conversational formatting for text-to-SQL.

The dataset (b-mc2/sql-create-context) has three columns:
  - question: natural-language question
  - context:  the CREATE TABLE statement(s) that define the schema
  - answer:   the gold SQL query

We convert each row into the *conversational* format SFTTrainer understands natively:
a `messages` list of {role, content}. SFTTrainer then applies the model's own chat
template at train time — which is critical: train-time formatting must match the
template the instruct model was pretrained with, or quality collapses.

We keep the system prompt identical between training and evaluation/inference. If you
change it, change it in ONE place (SYSTEM_PROMPT) so train/serve never drift.
"""

from __future__ import annotations

from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset

SYSTEM_PROMPT = (
    "You are a precise text-to-SQL assistant. Given a database schema and a question, "
    "respond with a single valid SQL query and nothing else."
)


def build_user_prompt(schema: str, question: str) -> str:
    """The user turn. Used identically at train, eval, and inference time."""
    return (
        f"### Database schema:\n{schema.strip()}\n\n"
        f"### Question:\n{question.strip()}\n\n"
        f"### SQL:"
    )


def _to_messages(example: dict) -> dict:
    """Map one raw row -> conversational messages for SFTTrainer."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(example["context"], example["question"])},
            {"role": "assistant", "content": example["answer"].strip()},
        ]
    }


def load_and_format(
    dataset_name: str,
    test_size: float = 0.05,
    seed: int = 42,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> DatasetDict:
    """Load the dataset, split it, and convert to the messages format.

    Returns a DatasetDict with `train` and `test` splits, each carrying:
      - messages:  for training
      - schema, question, gold_sql:  kept for evaluation (so eval doesn't re-parse messages)
    """
    raw = load_dataset(dataset_name, split="train")

    # Deterministic train/test split so eval is reproducible across runs.
    split = raw.train_test_split(test_size=test_size, seed=seed)

    def _prep(ds: Dataset, cap: Optional[int]) -> Dataset:
        if cap is not None:
            ds = ds.select(range(min(cap, len(ds))))
        # Keep raw fields for eval under stable names, then add messages.
        ds = ds.map(
            lambda ex: {
                "schema": ex["context"],
                "question": ex["question"],
                "gold_sql": ex["answer"].strip(),
                **_to_messages(ex),
            }
        )
        return ds

    return DatasetDict(
        train=_prep(split["train"], max_train_samples),
        test=_prep(split["test"], max_eval_samples),
    )


def preview(ds: DatasetDict, tokenizer, n: int = 2) -> None:
    """Print a few fully-templated training examples so you can eyeball correctness.

    Run this BEFORE training. You are checking that:
      1. the SQL appears in the assistant turn,
      2. the chat template wraps roles correctly (no missing/duplicated special tokens).
    """
    for i in range(n):
        text = tokenizer.apply_chat_template(
            ds["train"][i]["messages"], tokenize=False
        )
        print(f"\n===== EXAMPLE {i} (rendered with chat template) =====\n{text}")
