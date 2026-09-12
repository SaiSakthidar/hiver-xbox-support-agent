"""
FAISS-based retriever over Xbox threads.

Embeds customer messages with a sentence-transformer model, builds an index,
and at query time returns the top-k most similar (customer_text, brand_reply)
pairs to use as grounding context for the reply drafter.

The index is cached to disk so it only builds once.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

THREADS_CSV  = Path(__file__).parent.parent / "data" / "xbox_threads.csv"
INDEX_PATH   = Path(__file__).parent.parent / "outputs" / "faiss.index"
META_PATH    = Path(__file__).parent.parent / "outputs" / "faiss_meta.pkl"
EMBED_MODEL  = "all-MiniLM-L6-v2"   # 80 MB, fast, good enough
MAX_DOCS     = 15_000                # subsample so index stays manageable


def _load_model():
    from sentence_transformers import SentenceTransformer
    # local_files_only avoids a HuggingFace Hub network check on every load
    # when the model is already cached. Falls back to downloading if not cached.
    try:
        return SentenceTransformer(EMBED_MODEL, local_files_only=True)
    except Exception:
        return SentenceTransformer(EMBED_MODEL)


def build_index(force: bool = False) -> None:
    """Build and persist the FAISS index."""
    import faiss

    if INDEX_PATH.exists() and META_PATH.exists() and not force:
        print("Index already exists. Pass force=True to rebuild.")
        return

    INDEX_PATH.parent.mkdir(exist_ok=True)

    print("Loading threads…")
    df = pd.read_csv(THREADS_CSV).dropna(subset=["customer_text", "brand_reply"])
    df = df.head(MAX_DOCS)

    print(f"Encoding {len(df):,} customer messages…")
    model = _load_model()
    embeddings = model.encode(
        df["customer_text"].tolist(),
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product == cosine on L2-normed vecs
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    meta = df[["thread_id", "customer_text", "brand_reply", "escalated"]].to_dict("records")
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"Index saved → {INDEX_PATH}  ({index.ntotal:,} vectors, dim={dim})")


class Retriever:
    def __init__(self):
        import faiss

        if not INDEX_PATH.exists():
            print("Index not found — building now (one-time ~2 min)…")
            build_index()

        self._index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "rb") as f:
            self._meta = pickle.load(f)
        self._model = _load_model()

    def query(self, text: str, k: int = 3) -> list[dict]:
        """Return top-k similar historical threads."""
        vec = self._model.encode([text], normalize_embeddings=True)
        vec = np.array(vec, dtype="float32")
        scores, indices = self._index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = self._meta[idx].copy()
            row["similarity"] = float(score)
            results.append(row)
        return results


if __name__ == "__main__":
    build_index()
    r = Retriever()
    hits = r.query("my xbox won't connect to xbox live error code 0x80070102")
    for h in hits:
        print(f"\n[{h['similarity']:.3f}] {h['customer_text'][:80]}")
        print(f"  → {h['brand_reply'][:80]}")
