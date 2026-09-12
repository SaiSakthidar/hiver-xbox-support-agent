"""
Samples ~200 Xbox threads for the golden evaluation set using keyword-based
stratification across the 8 intent categories + a random slice for long-tail.

Output: eval/golden_unlabelled.csv  (ready to be labelled interactively)
"""

import re
import json
import pandas as pd
from pathlib import Path

THREADS_CSV = Path(__file__).parent.parent / "data" / "xbox_threads.csv"
OUT_CSV     = Path(__file__).parent.parent / "eval" / "golden_unlabelled.csv"
OUT_CSV.parent.mkdir(exist_ok=True)

N_PER_INTENT = 22
N_RANDOM     = 24
SEED         = 2024

# Keyword heuristics to pre-stratify (regex, case-insensitive)
INTENT_PATTERNS = {
    "account_access": [
        r"\bsign[- ]?in\b", r"\blogin\b", r"\blog[- ]?in\b", r"\bpassword\b",
        r"\bgamertag\b", r"\bemail\b.*\baccount\b", r"\baccount\b.*\bemail\b",
        r"\bcannot\s+access\b", r"\blocked\s+out\b", r"\bcan'?t\s+sign\b",
    ],
    "billing_subscription": [
        r"\brefund\b", r"\bcharge[sd]?\b", r"\bgift\s+card\b", r"\bcode\b.*\bcard\b",
        r"\bxbox\s+live\s+gold\b", r"\bgold\s+membership\b", r"\bspending\s+limit\b",
        r"\bpurchase\b", r"\bbill(ing)?\b", r"\bpayment\b", r"\bsubscri",
        r"\b\$\d+\b", r"\bprice\b", r"\bcost\b", r"\bcredit\s+card\b",
    ],
    "hardware_device": [
        r"\bcontroller\b", r"\bconsole\b", r"\bwon'?t\s+turn\s+on\b",
        r"\bhdmi\b", r"\bdisplay\b", r"\belite\s+controller\b",
        r"\bpower\s+(supply|brick|cord)\b", r"\brusset\s+ring\b",
        r"\bbroken\b", r"\bhardware\b", r"\bdisconnects?\b.*\bcontroller\b",
        r"\bcontroller\b.*\bdisconnects?\b", r"\boverheating\b",
    ],
    "network_connectivity": [
        r"\bxbox\s+live\b", r"\bconnect(ion|ivity|ing)?\b", r"\bnat\b",
        r"\bserver[s]?\b", r"\berror\s+code\b", r"\boffline\b",
        r"\binternet\b", r"\bwifi\b", r"\bwi-fi\b", r"\bping\b",
        r"\blatency\b", r"\bnetwork\b", r"\boutage\b", r"\bdown\b.*\bserver\b",
    ],
    "game_content": [
        r"\bgame\b", r"\bachievement[s]?\b", r"\bclip[s]?\b",
        r"\bbackwards?\s+compat", r"\bdlc\b", r"\bdownload(ing)?\b",
        r"\binstall(ing)?\b", r"\bgame\s+pass\b", r"\bplay(ing)?\b.*\bgame\b",
        r"\bgame\b.*\bupdate\b", r"\bpatch\b", r"\bsave\b.*\bgame\b",
    ],
    "app_feature": [
        r"\bapp\b", r"\bnetflix\b", r"\bhulu\b", r"\bpandora\b",
        r"\byoutube\b", r"\bmixer\b", r"\bbroadcast\b", r"\bstream(ing)?\b",
        r"\bfeature\s+request\b", r"\bfeedback\b", r"\bsuggestion\b",
        r"\bui\b", r"\binterface\b", r"\bupdate\s+app\b",
    ],
    "enforcement_ban": [
        r"\bban(ned)?\b", r"\bsuspend(ed)?\b", r"\benforcement\b",
        r"\bappeal\b", r"\breport(ed)?\b.*\bme\b", r"\bunfair\b",
        r"\bfraud(ulent)?\b", r"\bstolen\b.*\baccount\b", r"\bhack(ed)?\b",
    ],
    "other": [],  # catch-all, filled from random slice
}


def classify_by_keyword(text: str) -> str | None:
    """Returns first matching intent or None."""
    text_l = text.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        if intent == "other":
            continue
        if any(re.search(p, text_l) for p in patterns):
            return intent
    return None


def main():
    df = pd.read_csv(THREADS_CSV)
    df = df.dropna(subset=["customer_text", "brand_reply"])

    # Assign keyword-based tentative intent
    df["keyword_intent"] = df["customer_text"].apply(classify_by_keyword)

    frames = []
    used_ids = set()

    for intent, _ in INTENT_PATTERNS.items():
        if intent == "other":
            continue
        pool = df[df["keyword_intent"] == intent]
        n = min(N_PER_INTENT, len(pool))
        sample = pool.sample(n=n, random_state=SEED)
        sample = sample.copy()
        sample["suggested_intent"] = intent
        frames.append(sample)
        used_ids.update(sample["thread_id"].tolist())

    # Random slice from un-used rows (long-tail + "other")
    remaining = df[~df["thread_id"].isin(used_ids)]
    random_slice = remaining.sample(n=min(N_RANDOM, len(remaining)), random_state=SEED)
    random_slice = random_slice.copy()
    random_slice["suggested_intent"] = random_slice["keyword_intent"].fillna("other")
    frames.append(random_slice)

    golden = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=SEED)

    # Columns for human labeller to fill in
    golden["intent"] = ""              # human fills
    golden["should_escalate"] = ""     # human fills: yes / no
    golden["reply_quality"] = ""       # human fills: 1-5

    out_cols = [
        "thread_id", "customer_text", "brand_reply",
        "suggested_intent",            # hint from keyword (can override)
        "escalated",                   # our proxy label (for reference)
        "intent", "should_escalate", "reply_quality",
    ]
    golden[out_cols].to_csv(OUT_CSV, index=False)
    print(f"Saved {len(golden)} examples → {OUT_CSV}")
    print("\nIntent distribution (suggested):")
    print(golden["suggested_intent"].value_counts().to_string())


if __name__ == "__main__":
    main()
