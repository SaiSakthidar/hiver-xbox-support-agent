"""
Batch pipeline: runs the agent (or a baseline) over a CSV of customer messages
and writes predictions to outputs/.

Usage:
    python3 src/pipeline.py --input data/xbox_threads.csv --system agent --n 500
    python3 src/pipeline.py --input eval/golden_labelled.csv --system trivial
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).parent.parent))


import argparse
import json
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm

OUT_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def get_predict_fn(system: str):
    if system == "agent":
        from src.agent import run
        return run
    elif system == "trivial":
        from src.baselines import TrivialBaseline
        return TrivialBaseline().predict
    elif system == "simple":
        from src.baselines import SimpleBaseline
        b = SimpleBaseline()
        if not b.load():
            raise RuntimeError("SimpleBaseline not trained. Run: python3 src/baselines.py")
        return b.predict
    else:
        raise ValueError(f"Unknown system: {system}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/xbox_threads.csv")
    parser.add_argument("--system", choices=["agent", "trivial", "simple"], default="agent")
    parser.add_argument("--n",      type=int, default=None, help="Max rows to process")
    parser.add_argument("--out",    default=None, help="Output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.input).dropna(subset=["customer_text"])
    if args.n:
        df = df.head(args.n)

    predict_fn = get_predict_fn(args.system)
    out_path   = args.out or str(OUT_DIR / f"predictions_{args.system}.csv")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=args.system):
        pred = predict_fn(row["customer_text"])
        pred["thread_id"] = row.get("thread_id", "")
        pred["customer_text"] = row["customer_text"]
        results.append(pred)
        time.sleep(0.05)

    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved {len(results)} predictions → {out_path}")


if __name__ == "__main__":
    main()
