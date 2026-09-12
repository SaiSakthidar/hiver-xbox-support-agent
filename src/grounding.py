"""
Grounding score: measures how much the drafted reply is semantically grounded
in the retrieved source examples vs. being hallucinated / generic.

Score = mean cosine similarity between the draft reply embedding and the
embeddings of the k retrieved source replies.

A high score (>0.5) means the draft closely follows historical Xbox reply style.
A low score (<0.3) suggests the LLM ignored the retrieved context.

Used in evaluate.py to compare RAG agent vs. baselines.
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).parent.parent))


import numpy as np
from src.retriever import _load_model


_model = None


def _get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def grounding_score(draft_reply: str, source_replies: list[str]) -> float:
    """
    Returns mean cosine similarity between draft and source replies.
    Range: 0.0 (no grounding) to 1.0 (identical).
    """
    if not source_replies or not draft_reply.strip():
        return 0.0

    model = _get_model()
    texts = [draft_reply] + source_replies
    embeddings = model.encode(texts, normalize_embeddings=True)

    draft_vec = embeddings[0]
    source_vecs = embeddings[1:]
    similarities = source_vecs @ draft_vec
    return float(np.mean(similarities))


def grounding_score_from_thread_ids(
    draft_reply: str,
    source_thread_ids: list[str],
    threads_lookup: dict,
) -> float:
    """Convenience wrapper when you have thread IDs instead of reply texts."""
    source_replies = [
        threads_lookup[tid]["brand_reply"]
        for tid in source_thread_ids
        if tid in threads_lookup
    ]
    return grounding_score(draft_reply, source_replies)


if __name__ == "__main__":
    import pandas as pd
    from src.retriever import Retriever

    df = pd.read_csv("data/xbox_threads.csv")
    lookup = {str(r["thread_id"]): r for _, r in df.iterrows()}
    retriever = Retriever()

    test_cases = [
        "my xbox won't connect to xbox live, error code 0x87DD0006",
        "I need a refund for a game I accidentally purchased",
        "my controller keeps disconnecting randomly",
    ]

    print(f"{'Message':<55} {'Grounding':>10}")
    print("─" * 67)
    for msg in test_cases:
        from src.agent import draft_reply
        result = draft_reply(msg, "network_connectivity")
        hits = retriever.query(msg, k=3)
        source_replies = [h["brand_reply"] for h in hits]
        score = grounding_score(result["reply"], source_replies)
        print(f"{msg[:54]:<55} {score:>10.3f}")
