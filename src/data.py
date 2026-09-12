"""
Thread reconstruction for XboxSupport from the Twitter Customer Support dataset.

Builds (customer_message, brand_reply, thread_context) triples by walking
the response_tweet_id / in_response_to_tweet_id graph.

Key issues handled:
- Multi-tweet replies: Xbox splits long replies across numbered tweets with
  agent initials (^EZ). We stitch these into one reply and strip the suffix.
- Redacted user IDs: @115712-style handles are anonymised; we normalise them
  to @USER so the LLM doesn't treat them as real accounts.
- First-turn detection: customer first messages have NaN in_response_to_tweet_id
  relative to the thread, so we walk the graph forward from brand replies.
"""

import re
import pandas as pd
from pathlib import Path

RAW_CSV = Path(__file__).parent.parent / "data" / "twcs" / "twcs.csv"
XBOX_OUT = Path(__file__).parent.parent / "data" / "xbox_threads.csv"
BRAND = "XboxSupport"


def _redact_users(text: str) -> str:
    """Replace anonymised numeric handles (@123456) with @USER."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"@\d+", "@USER", text)


def _strip_agent_suffix(text: str) -> str:
    """Remove trailing tweet-number + agent-initial suffix like ' 2 ^JL'."""
    return re.sub(r"\s+\d+\s+\^[A-Z]{2,3}\s*$", "", text).strip()


def _stitch_brand_tweets(tweets: list[str]) -> str:
    """
    Stitch a sequence of brand tweets that form one logical reply.
    Xbox uses ' 1 ^XX ... 2 ^XX ... 3 ^XX' across consecutive tweets.
    """
    parts = []
    for t in tweets:
        clean = _strip_agent_suffix(_redact_users(t))
        if clean:
            parts.append(clean)
    return " ".join(parts)


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV, dtype={"tweet_id": str, "in_response_to_tweet_id": str,
                                        "response_tweet_id": str})


def build_xbox_threads(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      thread_id       - ID of the first customer tweet in the thread
      customer_text   - cleaned customer message (first turn)
      brand_reply     - stitched, cleaned brand reply
      prior_context   - any preceding turns (list of dicts) as JSON string
      escalated       - bool: brand directed customer to DM / chat / provide ID
    """
    if df is None:
        df = load_raw()

    df = df.copy()
    df["tweet_id"] = df["tweet_id"].astype(str)
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].astype(str)
    df["response_tweet_id"] = df["response_tweet_id"].astype(str)

    # Index by tweet_id for O(1) lookup
    id_to_row = df.set_index("tweet_id").to_dict("index")

    xbox_out = df[(df["author_id"] == BRAND) & (df["inbound"] == False)].copy()

    records = []
    seen_threads = set()

    for _, brand_row in xbox_out.iterrows():
        # Walk back to find the customer message this brand tweet responds to
        resp_to = brand_row["in_response_to_tweet_id"]
        if resp_to == "nan" or pd.isna(resp_to):
            continue

        cust_row = id_to_row.get(str(resp_to))
        if cust_row is None:
            continue
        if not cust_row.get("inbound", False):
            continue  # brand replied to itself (multi-part chain start)

        customer_tweet_id = str(resp_to)
        if customer_tweet_id in seen_threads:
            continue
        seen_threads.add(customer_tweet_id)

        # Collect all consecutive brand reply parts for this customer tweet
        # by following response_tweet_id forward
        brand_parts = [brand_row["text"]]
        next_resp = str(brand_row.get("response_tweet_id", "nan"))
        visited = {brand_row["tweet_id"]}
        while next_resp != "nan" and next_resp not in visited:
            next_row = id_to_row.get(next_resp)
            if next_row is None:
                break
            if next_row.get("author_id") != BRAND or next_row.get("inbound", True):
                break
            brand_parts.append(next_row["text"])
            visited.add(next_resp)
            next_resp = str(next_row.get("response_tweet_id", "nan"))

        brand_reply = _stitch_brand_tweets(brand_parts)

        # Collect prior context: walk back from customer tweet
        prior = []
        walk_id = str(cust_row.get("in_response_to_tweet_id", "nan"))
        depth = 0
        while walk_id != "nan" and depth < 6:
            prev = id_to_row.get(walk_id)
            if prev is None:
                break
            prior.insert(0, {
                "author": prev.get("author_id"),
                "text": _redact_users(str(prev.get("text", ""))),
                "inbound": prev.get("inbound"),
            })
            walk_id = str(prev.get("in_response_to_tweet_id", "nan"))
            depth += 1

        # Escalation proxy: brand directed to DM, chat, gamertag, or support URL
        escalation_patterns = [
            r"dm\b", r"direct\s+message", r"private\s+message",
            r"chat\s+(team|support|us)", r"contact\s+us",
            r"gamertag", r"support\.xbox\.com", r"xbox\.com/support",
            r"phone\s+support", r"call\s+us",
        ]
        escalated = any(
            re.search(p, brand_reply.lower()) for p in escalation_patterns
        )

        records.append({
            "thread_id": customer_tweet_id,
            "customer_text": _redact_users(str(cust_row.get("text", ""))),
            "brand_reply": brand_reply,
            "prior_context": str(prior),
            "escalated": escalated,
        })

    result = pd.DataFrame(records)
    return result


def main():
    print("Loading raw data…")
    df = load_raw()
    print(f"  Total tweets: {len(df):,}")

    print("Building Xbox threads…")
    threads = build_xbox_threads(df)
    print(f"  Threads extracted: {len(threads):,}")
    print(f"  Escalated: {threads['escalated'].sum():,} ({threads['escalated'].mean():.1%})")

    threads.to_csv(XBOX_OUT, index=False)
    print(f"  Saved → {XBOX_OUT}")

    print("\nSample:")
    for _, row in threads.head(3).iterrows():
        print(f"\n  Customer : {row['customer_text'][:100]}")
        print(f"  Reply    : {row['brand_reply'][:100]}")
        print(f"  Escalated: {row['escalated']}")


if __name__ == "__main__":
    main()
