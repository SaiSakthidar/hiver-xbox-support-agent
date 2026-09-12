"""
Context ablation: measures how much context-awareness improves intent accuracy.

Runs the agent on golden-labelled examples that have prior conversation context,
once with context and once without, then compares accuracy.

Usage:
    python3 src/ablation.py
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

GOLDEN      = Path(__file__).parent.parent / "eval" / "golden_labelled.csv"
THREADS_CSV = Path(__file__).parent.parent / "data" / "xbox_threads.csv"
RESULTS_DIR = Path(__file__).parent.parent / "outputs" / "eval_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _has_context(x) -> bool:
    return str(x).strip() not in ("[]", "nan", "", "None")


def main():
    from src.agent import classify_intent

    golden = pd.read_csv(GOLDEN)
    golden = golden[golden["intent"].notna() & (golden["intent"].str.strip() != "")]

    threads = pd.read_csv(THREADS_CSV)[["thread_id", "prior_context"]]
    merged = golden.merge(threads, on="thread_id", how="left")

    ctx_df = merged[merged["prior_context"].apply(_has_context)].copy()
    noctx_df = merged[~merged["prior_context"].apply(_has_context)].copy()

    print(f"Examples WITH prior context : {len(ctx_df)}")
    print(f"Examples WITHOUT prior context: {len(noctx_df)}")

    if len(ctx_df) < 5:
        print("Too few context examples for a reliable ablation.")
        return

    with_preds, without_preds, labels = [], [], []
    total = len(ctx_df)

    for i, (_, row) in enumerate(ctx_df.iterrows(), 1):
        print(f"  [{i}/{total}] classifying...", end="\r")
        labels.append(row["intent"])

        r_with    = classify_intent(row["customer_text"], prior_context=row["prior_context"])
        r_without = classify_intent(row["customer_text"], prior_context=None)

        with_preds.append(r_with["intent"])
        without_preds.append(r_without["intent"])

    print()

    acc_with    = accuracy_score(labels, with_preds)
    acc_without = accuracy_score(labels, without_preds)
    f1_with     = f1_score(labels, with_preds,    average="macro", zero_division=0)
    f1_without  = f1_score(labels, without_preds, average="macro", zero_division=0)

    print(f"\n{'='*60}")
    print(f"  Context ablation  (n={len(ctx_df)} examples with prior turns)")
    print(f"{'='*60}")
    print(f"  {'Metric':<22} {'With context':>13} {'No context':>11} {'Delta':>8}")
    print(f"  {'-'*56}")
    print(f"  {'Intent accuracy':<22} {acc_with:>13.3f} {acc_without:>11.3f} {acc_with - acc_without:>+8.3f}")
    print(f"  {'Intent macro-F1':<22} {f1_with:>13.3f} {f1_without:>11.3f} {f1_with - f1_without:>+8.3f}")
    print()

    results = {
        "n_with_context":         len(ctx_df),
        "n_without_context":      len(noctx_df),
        "accuracy_with_context":  round(acc_with, 4),
        "accuracy_no_context":    round(acc_without, 4),
        "accuracy_delta":         round(acc_with - acc_without, 4),
        "macro_f1_with_context":  round(f1_with, 4),
        "macro_f1_no_context":    round(f1_without, 4),
        "macro_f1_delta":         round(f1_with - f1_without, 4),
    }

    out = RESULTS_DIR / "ablation_context.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out}")
    return results


if __name__ == "__main__":
    main()
