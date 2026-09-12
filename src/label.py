"""
Interactive CLI labelling tool for the golden evaluation set.

Usage:
    python3 src/label.py

Controls:
  - Type the intent number (or name) and press Enter
  - Then answer escalate: y/n
  - Then rate reply quality: 1-5
  - Type 'q' at any point to save progress and quit
  - Type 'b' to go back one example
  - Type 's' to skip (leaves blank, revisit later)

Progress is saved after every label so you can stop and resume.
"""

import pandas as pd
from pathlib import Path

UNLABELLED = Path(__file__).parent.parent / "eval" / "golden_unlabelled.csv"
LABELLED   = Path(__file__).parent.parent / "eval" / "golden_labelled.csv"

INTENTS = [
    "account_access",
    "billing_subscription",
    "hardware_device",
    "network_connectivity",
    "game_content",
    "app_feature",
    "enforcement_ban",
    "other",
]

INTENT_MENU = "\n".join(f"  {i+1}. {name}" for i, name in enumerate(INTENTS))


def _clr():
    print("\033[2J\033[H", end="")


def _bold(s):
    return f"\033[1m{s}\033[0m"


def _dim(s):
    return f"\033[2m{s}\033[0m"


def _ask(prompt, valid=None):
    while True:
        val = input(prompt).strip().lower()
        if val == "q":
            return "q"
        if val == "b":
            return "b"
        if val == "s":
            return "s"
        if valid is None or val in valid:
            return val
        print(f"  Please enter one of: {valid}")


def resolve_intent(raw: str) -> str | None:
    """Accept '1'-'8' or partial intent name."""
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(INTENTS):
            return INTENTS[idx]
    for name in INTENTS:
        if raw in name:
            return name
    return None


def main():
    if not UNLABELLED.exists():
        print(f"Run  python3 src/sample_golden.py  first to generate {UNLABELLED}")
        return

    df = pd.read_csv(UNLABELLED, dtype=str).fillna("")

    # Load existing progress if any
    if LABELLED.exists():
        done = pd.read_csv(LABELLED, dtype=str).fillna("")
        labelled_ids = set(done["thread_id"].tolist())
    else:
        done = pd.DataFrame(columns=df.columns)
        labelled_ids = set()

    total = len(df)
    to_label = df[~df["thread_id"].isin(labelled_ids)].reset_index(drop=True)
    print(f"Progress: {len(labelled_ids)}/{total} labelled. {len(to_label)} remaining.")

    completed = []
    i = 0
    while i < len(to_label):
        row = to_label.iloc[i]
        _clr()
        pct = int((len(labelled_ids) + i) / total * 100)
        print(_bold(f"─── Example {len(labelled_ids)+i+1}/{total}  ({pct}%) ───────────────────────"))
        print()
        print(_bold("Customer:"))
        print(f"  {row['customer_text']}")
        print()
        print(_bold("Brand reply:"))
        print(f"  {row['brand_reply']}")
        print()
        print(_dim(f"Suggested intent : {row['suggested_intent']}"))
        print(_dim(f"Proxy escalated  : {row['escalated']}"))
        print()
        print("Intents:")
        print(INTENT_MENU)
        print()

        # Intent
        raw = _ask("Intent [1-8 or name, b=back, s=skip, q=quit]: ", None)
        if raw == "q":
            break
        if raw == "b" and i > 0:
            # undo last completed
            if completed:
                labelled_ids.discard(completed[-1]["thread_id"])
                completed.pop()
            i -= 1
            continue
        if raw == "s":
            i += 1
            continue

        intent = resolve_intent(raw)
        if not intent:
            print("  Unknown intent, try again.")
            continue

        # Escalate
        esc_raw = _ask("Should escalate? [y/n, b, s, q]: ", {"y", "n"})
        if esc_raw == "q":
            break
        if esc_raw == "b":
            continue
        if esc_raw == "s":
            i += 1
            continue

        # Reply quality
        qual_raw = _ask("Reply quality 1-5 [1=terrible, 5=excellent, b, s, q]: ", {"1","2","3","4","5"})
        if qual_raw == "q":
            break
        if qual_raw == "b":
            continue
        if qual_raw == "s":
            i += 1
            continue

        entry = row.to_dict()
        entry["intent"] = intent
        entry["should_escalate"] = "yes" if esc_raw == "y" else "no"
        entry["reply_quality"] = qual_raw
        completed.append(entry)
        labelled_ids.add(row["thread_id"])
        i += 1

    # Save
    if completed:
        new_df = pd.DataFrame(completed)
        result = pd.concat([done, new_df], ignore_index=True)
        result.to_csv(LABELLED, index=False)
        print(f"\nSaved {len(result)} labelled examples → {LABELLED}")
    else:
        print("\nNo new labels saved.")

    labelled_now = len(labelled_ids)
    print(f"Total labelled: {labelled_now}/{total}  ({labelled_now/total:.0%})")
    if labelled_now < 150:
        print(f"  Need at least 150. {150 - labelled_now} more to go!")
    elif labelled_now < 200:
        print(f"  Good progress! Aim for 200 for stronger eval.")
    else:
        print("  Golden set complete!")


if __name__ == "__main__":
    main()
