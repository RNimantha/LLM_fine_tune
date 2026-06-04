"""Evaluate a text-to-SQL model with metrics that actually mean something.

Three metrics, in order of credibility:
  1. parse_rate   — % of generated strings that are valid, parseable SQL (sqlglot).
  2. exact_match  — normalized string equality (lowercased, whitespace-collapsed).
  3. ast_match    — sqlglot parses gold + pred into ASTs and compares them. Robust to
                    formatting, alias casing, and trivial reordering. THIS is your headline.

Run base vs fine-tuned to get a comparison table:
    python -m src.evaluate --model Qwen/Qwen2.5-1.5B-Instruct --limit 300   # base
    python -m src.evaluate --model outputs/qwen2.5-1.5b-text2sql-qlora --limit 300  # tuned

Execution accuracy (running queries against a real DB) is the gold standard but needs
per-schema databases; it's the Week-6 extension in CURRICULUM.md.
"""

from __future__ import annotations

import argparse
import re

import sqlglot
from sqlglot import exp

from src.config import load_config
from src.data import load_and_format
from src.infer import generate_sql, load_model_and_tokenizer


def normalize(sql: str) -> str:
    """Cheap normalization for exact-match: lowercase, collapse whitespace, drop trailing ;."""
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql.lower()


def parses(sql: str) -> bool:
    try:
        sqlglot.parse_one(sql)
        return True
    except Exception:
        return False


def ast_equal(gold: str, pred: str) -> bool:
    """Compare two queries by their canonical ASTs.

    sqlglot's normalized SQL string is a stable serialization of the parsed AST, so
    equal canonical strings => structurally equivalent queries.
    """
    try:
        g = sqlglot.parse_one(gold).sql(normalize=True, pretty=False)
        p = sqlglot.parse_one(pred).sql(normalize=True, pretty=False)
        return g.lower() == p.lower()
    except Exception:
        return False


def evaluate(model_path: str, config_path: str, limit: int | None) -> dict:
    cfg = load_config(config_path)
    ds = load_and_format(
        dataset_name=cfg.data.dataset_name,
        test_size=cfg.data.test_size,
        seed=cfg.data.seed,
        max_eval_samples=limit,
    )["test"]

    model, tokenizer = load_model_and_tokenizer(model_path)

    n = len(ds)
    n_parse = n_exact = n_ast = 0
    failures = []

    for i, row in enumerate(ds):
        pred = generate_sql(model, tokenizer, row["schema"], row["question"])
        gold = row["gold_sql"]

        ok_parse = parses(pred)
        ok_exact = normalize(pred) == normalize(gold)
        ok_ast = ast_equal(gold, pred)

        n_parse += ok_parse
        n_exact += ok_exact
        n_ast += ok_ast

        if not ok_ast and len(failures) < 10:
            failures.append({"question": row["question"], "gold": gold, "pred": pred})

        if (i + 1) % 25 == 0:
            print(f"[{i+1}/{n}] parse={n_parse/(i+1):.2%} "
                  f"exact={n_exact/(i+1):.2%} ast={n_ast/(i+1):.2%}")

    results = {
        "model": model_path,
        "n": n,
        "parse_rate": n_parse / n,
        "exact_match": n_exact / n,
        "ast_match": n_ast / n,
        "sample_failures": failures,
    }
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="base model id, adapter dir, or merged dir")
    p.add_argument("--config", default="configs/qlora_qwen.yaml")
    p.add_argument("--limit", type=int, default=300, help="number of test examples")
    args = p.parse_args()

    res = evaluate(args.model, args.config, args.limit)
    print("\n================ RESULTS ================")
    print(f"model       : {res['model']}")
    print(f"n examples  : {res['n']}")
    print(f"parse_rate  : {res['parse_rate']:.2%}")
    print(f"exact_match : {res['exact_match']:.2%}")
    print(f"ast_match   : {res['ast_match']:.2%}  <-- headline metric")
    print("\n--- sample failures (gold vs pred) ---")
    for f in res["sample_failures"][:5]:
        print(f"\nQ: {f['question']}\nGOLD: {f['gold']}\nPRED: {f['pred']}")


if __name__ == "__main__":
    main()
