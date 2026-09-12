"""
Two baselines to compare against the LLM agent:

  Baseline 1 — Trivial:
    Intent: always predict the majority class ("game_content")
    Reply : fixed canned response
    Escalate: always "auto"

  Baseline 2 — Simple (TF-IDF + LogReg + nearest-neighbour reply):
    Intent: TF-IDF vectoriser + Logistic Regression, trained on labelled golden set
    Reply : cosine-nearest-neighbour retrieval from xbox_threads (TF-IDF vectors)
    Escalate: keyword rule-based

Trained on train split of golden_labelled.csv; evaluated on test split.
Run standalone to train and persist models.
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).parent.parent))


import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

LABELLED_CSV = Path(__file__).parent.parent / "eval" / "golden_labelled.csv"
THREADS_CSV  = Path(__file__).parent.parent / "data" / "xbox_threads.csv"
MODEL_DIR    = Path(__file__).parent.parent / "outputs"
MODEL_DIR.mkdir(exist_ok=True)

TFIDF_CLF_PATH  = MODEL_DIR / "tfidf_clf.pkl"
TFIDF_RET_PATH  = MODEL_DIR / "tfidf_retriever.pkl"
MAJORITY_CLASS  = "game_content"
CANNED_REPLY    = (
    "@USER Thanks for reaching out! Please DM us with your gamertag "
    "so we can look into this for you."
)

ESCALATION_KEYWORDS = re.compile(
    r"\b(ban(ned)?|suspend(ed)?|fraud|stolen|hack(ed)?|refund|charge[sd]?|"
    r"hardware|warranty|broken|won'?t\s+turn\s+on|damaged)\b",
    re.I,
)


# ── Baseline 1: Trivial ────────────────────────────────────────────────────────

class TrivialBaseline:
    name = "trivial"

    def predict(self, customer_text: str) -> dict:
        return {
            "intent": MAJORITY_CLASS,
            "confidence": "high",
            "reply": CANNED_REPLY,
            "decision": "auto",
            "reason": "Trivial baseline always auto-handles.",
        }


# ── Baseline 2: TF-IDF + LogReg ───────────────────────────────────────────────

class SimpleBaseline:
    name = "simple"

    def __init__(self):
        self._clf = None
        self._ret_vecs = None
        self._ret_meta = None
        self._tfidf_clf = None
        self._tfidf_ret = None

    def _clean(self, text: str) -> str:
        text = re.sub(r"@\S+|https?://\S+", " ", str(text))
        text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
        return " ".join(text.split())

    def train(self, train_df: pd.DataFrame, threads_df: pd.DataFrame) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        X = train_df["customer_text"].apply(self._clean).tolist()
        y = train_df["intent"].tolist()

        self._tfidf_clf = TfidfVectorizer(ngram_range=(1, 2), max_features=30_000)
        X_vec = self._tfidf_clf.fit_transform(X)
        self._clf = LogisticRegression(max_iter=500, C=1.0)
        self._clf.fit(X_vec, y)

        # Build TF-IDF retrieval index over all threads
        ret_texts = threads_df["customer_text"].apply(self._clean).tolist()
        self._tfidf_ret = TfidfVectorizer(ngram_range=(1, 2), max_features=30_000)
        self._ret_vecs = self._tfidf_ret.fit_transform(ret_texts)
        self._ret_meta = threads_df[["customer_text", "brand_reply"]].to_dict("records")

        with open(TFIDF_CLF_PATH, "wb") as f:
            pickle.dump((self._tfidf_clf, self._clf), f)
        with open(TFIDF_RET_PATH, "wb") as f:
            pickle.dump((self._tfidf_ret, self._ret_vecs, self._ret_meta), f)
        print(f"SimpleBaseline trained on {len(X)} examples.")

    def load(self) -> bool:
        if not (TFIDF_CLF_PATH.exists() and TFIDF_RET_PATH.exists()):
            return False
        with open(TFIDF_CLF_PATH, "rb") as f:
            self._tfidf_clf, self._clf = pickle.load(f)
        with open(TFIDF_RET_PATH, "rb") as f:
            self._tfidf_ret, self._ret_vecs, self._ret_meta = pickle.load(f)
        return True

    def _retrieve_reply(self, text: str) -> str:
        from sklearn.metrics.pairwise import cosine_similarity

        vec = self._tfidf_ret.transform([self._clean(text)])
        sims = cosine_similarity(vec, self._ret_vecs).flatten()
        idx = int(np.argmax(sims))
        return self._ret_meta[idx]["brand_reply"]

    def predict(self, customer_text: str) -> dict:
        if self._clf is None:
            if not self.load():
                raise RuntimeError("SimpleBaseline not trained. Run baselines.py first.")

        vec = self._tfidf_clf.transform([self._clean(customer_text)])
        intent = self._clf.predict(vec)[0]

        reply = self._retrieve_reply(customer_text)

        escalated = bool(ESCALATION_KEYWORDS.search(customer_text))
        decision = "escalate" if escalated else "auto"
        reason = "Keyword rule triggered escalation." if escalated else "No escalation keywords found."

        return {
            "intent": intent,
            "confidence": "medium",
            "reply": reply,
            "decision": decision,
            "reason": reason,
        }


# ── Training entry point ───────────────────────────────────────────────────────

def train_simple_baseline(test_size: float = 0.2) -> None:
    if not LABELLED_CSV.exists():
        raise FileNotFoundError(f"Golden set not found: {LABELLED_CSV}. Label first.")

    labelled = pd.read_csv(LABELLED_CSV).dropna(subset=["intent", "customer_text"])
    labelled = labelled[labelled["intent"].str.strip() != ""]

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(labelled, test_size=test_size, random_state=42,
                                          stratify=labelled["intent"])
    # Save split for eval harness
    train_df.to_csv(MODEL_DIR / "golden_train.csv", index=False)
    test_df.to_csv(MODEL_DIR / "golden_test.csv", index=False)

    threads = pd.read_csv(THREADS_CSV).dropna(subset=["customer_text", "brand_reply"])

    sb = SimpleBaseline()
    sb.train(train_df, threads)

    # Quick accuracy check
    from sklearn.metrics import classification_report
    X_test = test_df["customer_text"].apply(sb._clean).tolist()
    y_true = test_df["intent"].tolist()
    X_vec  = sb._tfidf_clf.transform(X_test)
    y_pred = sb._clf.predict(X_vec)
    print("\nSimpleBaseline test accuracy:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    train_simple_baseline()
