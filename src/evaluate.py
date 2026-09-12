"""
Evaluation harness for the XboxSupport AI agent.

Metrics computed:
  Intent classification:
    - Accuracy, macro-F1, per-class F1
  Escalation:
    - Accuracy, precision, recall, F1 vs human labels
  Reply quality (LLM-as-judge):
    - Mean score (1-5) per system
    - Cohen's kappa between judge scores and human reply_quality ratings

Usage:
    python3 src/evaluate.py --system agent     # evaluate LLM agent
    python3 src/evaluate.py --system trivial   # evaluate trivial baseline
    python3 src/evaluate.py --system simple    # evaluate TF-IDF baseline
    python3 src/evaluate.py --all              # run all three and compare
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).parent.parent))


import argparse
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

GOLDEN_TEST     = Path(__file__).parent.parent / "outputs" / "golden_test.csv"
GOLDEN_LABELLED = Path(__file__).parent.parent / "eval" / "golden_labelled.csv"
RESULTS_DIR  = Path(__file__).parent.parent / "outputs" / "eval_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Judge rubric ──────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are an expert evaluator of customer support replies.

Rate the given reply to the customer message on a scale of 1-5:
  5 = Excellent: empathetic, actionable, addresses the issue, concise
  4 = Good: addresses the issue but could be more specific or warm
  3 = Acceptable: partially relevant, generic but not wrong
  2 = Poor: misses the issue or is unhelpful
  1 = Terrible: wrong, rude, or completely off-topic

Reply with JSON only: {"score": <integer 1-5>, "reason": "<one sentence>"}"""


def llm_judge_score(customer_text: str, reply: str) -> dict:
    from src.llm import chat

    prompt = f'Customer: """{customer_text}"""\n\nReply: """{reply}"""'
    raw = chat(prompt, system=JUDGE_SYSTEM, max_tokens=100, temperature=0.0)
    try:
        result = json.loads(raw.strip())
        score = int(result.get("score", 3))
        score = max(1, min(5, score))
        return {"score": score, "reason": result.get("reason", "")}
    except Exception:
        return {"score": 3, "reason": "parse error"}


# ── Cohen's kappa ─────────────────────────────────────────────────────────────

def cohens_kappa(rater1: list[int], rater2: list[int]) -> float:
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(rater1, rater2)


# ── Run a system over the test set ────────────────────────────────────────────

def run_system(system_name: str, test_df: pd.DataFrame) -> pd.DataFrame:
    """Run agent/baseline over test_df, return df with predictions."""
    if system_name == "agent":
        from src.agent import run as agent_run
        predict_fn = lambda text: agent_run(text)
    elif system_name == "trivial":
        from src.baselines import TrivialBaseline
        b = TrivialBaseline()
        predict_fn = b.predict
    elif system_name == "simple":
        from src.baselines import SimpleBaseline
        b = SimpleBaseline()
        predict_fn = b.predict
    else:
        raise ValueError(f"Unknown system: {system_name}")

    rows = []
    total = len(test_df)
    for i, (_, row) in enumerate(test_df.iterrows(), 1):
        print(f"  [{i}/{total}] {system_name} ...", end="\r")
        pred = predict_fn(row["customer_text"])
        rows.append({
            "thread_id":      row["thread_id"],
            "customer_text":  row["customer_text"],
            "true_intent":    row["intent"],
            "true_escalate":  row["should_escalate"],
            "human_quality":  row.get("reply_quality", np.nan),
            "pred_intent":    pred.get("intent", "other"),
            "pred_decision":  pred.get("decision", "auto"),
            "pred_reply":     pred.get("reply", ""),
            "pred_reason":    pred.get("reason", ""),
        })
        time.sleep(0.1)   # gentle rate limiting

    return pd.DataFrame(rows)


# ── Score predictions ─────────────────────────────────────────────────────────

def score(pred_df: pd.DataFrame, run_judge: bool = True) -> dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report,
        precision_score, recall_score,
    )

    metrics = {}

    # Intent
    y_true_i = pred_df["true_intent"].tolist()
    y_pred_i = pred_df["pred_intent"].tolist()
    metrics["intent_accuracy"] = accuracy_score(y_true_i, y_pred_i)
    metrics["intent_macro_f1"] = f1_score(y_true_i, y_pred_i, average="macro", zero_division=0)
    metrics["intent_report"]   = classification_report(y_true_i, y_pred_i, zero_division=0)

    # Escalation
    esc_df = pred_df[pred_df["true_escalate"].isin(["yes", "no"])].copy()
    if len(esc_df) > 0:
        y_true_e = (esc_df["true_escalate"] == "yes").astype(int).tolist()
        y_pred_e = (esc_df["pred_decision"] == "escalate").astype(int).tolist()
        metrics["escalation_accuracy"]  = accuracy_score(y_true_e, y_pred_e)
        metrics["escalation_precision"] = precision_score(y_true_e, y_pred_e, zero_division=0)
        metrics["escalation_recall"]    = recall_score(y_true_e, y_pred_e, zero_division=0)
        metrics["escalation_f1"]        = f1_score(y_true_e, y_pred_e, zero_division=0)

    # LLM judge on agent's predicted replies (max 50 to control cost)
    if run_judge:
        sample = pred_df.sample(min(50, len(pred_df)), random_state=42)
        judge_pred_scores = []
        for _, row in sample.iterrows():
            j = llm_judge_score(row["customer_text"], row["pred_reply"])
            judge_pred_scores.append(j["score"])
        metrics["judge_mean_score"] = float(np.mean(judge_pred_scores))

        # ── Judge-human calibration ────────────────────────────────────────────
        # Run judge on the SAME historical brand replies the human already rated.
        # Comparing judge vs human on identical content gives true calibration.
        golden_test_path = Path(__file__).parent.parent / "outputs" / "golden_test.csv"
        calib_df = pd.read_csv(golden_test_path) if golden_test_path.exists() else pred_df
        calib_sample = calib_df.dropna(subset=["reply_quality"]).copy()
        calib_sample = calib_sample[
            calib_sample["reply_quality"].astype(str).str.strip() != ""
        ]
        # brand_reply comes from the test CSV if present, else fall back to pred_reply
        reply_col = "brand_reply" if "brand_reply" in calib_sample.columns else "pred_reply"
        calib_sample = calib_sample.head(50)

        judge_calib_scores = []
        human_calib_scores = []
        for _, row in calib_sample.iterrows():
            try:
                h = int(float(row["reply_quality"]))
            except (ValueError, TypeError):
                continue
            j = llm_judge_score(row["customer_text"], row[reply_col])
            judge_calib_scores.append(j["score"])
            human_calib_scores.append(h)

        if len(judge_calib_scores) >= 10:
            metrics["judge_human_kappa"] = cohens_kappa(judge_calib_scores, human_calib_scores)
            metrics["judge_human_pearson_r"] = float(
                pearsonr(judge_calib_scores, human_calib_scores).statistic
            )
            metrics["judge_calibration_n"] = len(judge_calib_scores)

    return metrics


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(system_name: str, metrics: dict) -> None:
    print(f"\n{'═'*60}")
    print(f"  System: {system_name.upper()}")
    print(f"{'═'*60}")
    print(f"  Intent accuracy : {metrics.get('intent_accuracy', 0):.3f}")
    print(f"  Intent macro-F1 : {metrics.get('intent_macro_f1', 0):.3f}")
    print(f"  Escalation acc  : {metrics.get('escalation_accuracy', 0):.3f}")
    print(f"  Escalation F1   : {metrics.get('escalation_f1', 0):.3f}")
    print(f"  Judge mean score: {metrics.get('judge_mean_score', 0):.2f} / 5")
    if "judge_human_kappa" in metrics:
        print(f"  Judge-Human κ   : {metrics['judge_human_kappa']:.3f}")
        print(f"  Judge-Human r   : {metrics['judge_human_pearson_r']:.3f}")
    print(f"\nIntent classification report:")
    print(metrics.get("intent_report", ""))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["agent", "trivial", "simple"], default="agent")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge (faster)")
    args = parser.parse_args()

    if not GOLDEN_TEST.exists():
        raise FileNotFoundError(
            f"{GOLDEN_TEST} not found.\n"
            "Run:  python3 src/baselines.py  (to split golden_labelled.csv into train/test)"
        )

    # Baselines use 20% test split (can't use train set — they trained on it).
    # LLM agent is zero-shot, so it can be evaluated on the full labelled set
    # without data leakage — gives more reliable metrics.
    baseline_test_df = pd.read_csv(GOLDEN_TEST)
    agent_test_df = (
        pd.read_csv(GOLDEN_LABELLED)
        if GOLDEN_LABELLED.exists()
        else baseline_test_df
    )
    agent_test_df = agent_test_df[
        agent_test_df["intent"].notna() &
        (agent_test_df["intent"].str.strip() != "")
    ]

    systems = ["trivial", "simple", "agent"] if args.all else [args.system]

    all_metrics = {}
    for sys in systems:
        test_df = agent_test_df if sys == "agent" else baseline_test_df
        print(f"\nRunning {sys} on {len(test_df)} examples…")
        pred_df = run_system(sys, test_df)
        pred_df.to_csv(RESULTS_DIR / f"preds_{sys}.csv", index=False)

        metrics = score(pred_df, run_judge=not args.no_judge)
        all_metrics[sys] = metrics
        print_report(sys, metrics)

        with open(RESULTS_DIR / f"metrics_{sys}.json", "w") as f:
            json.dump({k: v for k, v in metrics.items() if k != "intent_report"}, f, indent=2)

    if args.all:
        print("\n\n── Summary ──────────────────────────────────────────────")
        print(f"{'System':<12} {'Intent Acc':>12} {'Intent F1':>10} {'Esc F1':>8} {'Judge':>8}")
        print("-" * 54)
        for sys, m in all_metrics.items():
            print(
                f"{sys:<12} {m.get('intent_accuracy',0):>12.3f}"
                f" {m.get('intent_macro_f1',0):>10.3f}"
                f" {m.get('escalation_f1',0):>8.3f}"
                f" {m.get('judge_mean_score',0):>8.2f}"
            )


if __name__ == "__main__":
    main()
