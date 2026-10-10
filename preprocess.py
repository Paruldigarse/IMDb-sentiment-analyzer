"""
preprocess.py
─────────────
Downloads the IMDB 50 K movie-review dataset (via sklearn or a local CSV),
cleans the raw text, and saves train/test splits plus the TF-IDF matrix.

Cleaning pipeline
  1. Lowercase
  2. Strip HTML tags
  3. Remove punctuation & digits
  4. Tokenise (whitespace)
  5. Remove English stop-words (NLTK)
  6. Rejoin tokens
"""

import os
import re
import string
import joblib
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_CSV  = os.path.join(DATA_DIR, "train.csv")
TEST_CSV   = os.path.join(DATA_DIR, "test.csv")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")


# ── NLTK assets ────────────────────────────────────────────────────────────
def ensure_nltk():
    for resource in ("stopwords", "punkt"):
        try:
            nltk.data.find(f"corpora/{resource}" if resource == "stopwords"
                           else f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


# ── text cleaner ───────────────────────────────────────────────────────────
_HTML_RE  = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")

def clean_text(text: str, stop_words: set) -> str:
    """Return a cleaned, stop-word-free string."""
    text = text.lower()
    text = _HTML_RE.sub(" ", text)                          # strip HTML
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punct
    text = re.sub(r"\d+", " ", text)                        # remove digits
    tokens = _SPACE_RE.sub(" ", text).strip().split()       # tokenise
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


# ── dataset loader ─────────────────────────────────────────────────────────
def load_imdb_dataframe() -> pd.DataFrame:
    """
    Try three strategies in order:
      1. data/IMDB Dataset.csv  (Kaggle CSV, col names: review / sentiment)
      2. data/aclImdb/          (original tar.gz extracted folder)
      3. Synthesise a tiny 200-row demo set for smoke-testing
    """
    kaggle_csv = os.path.join(DATA_DIR, "IMDB Dataset.csv")
    acl_dir    = os.path.join(DATA_DIR, "aclImdb")

    # ── Strategy 1: Kaggle CSV ────────────────────────────────────────────
    if os.path.exists(kaggle_csv):
        print(f"[preprocess] Loading Kaggle CSV → {kaggle_csv}")
        df = pd.read_csv(kaggle_csv)
        df.columns = df.columns.str.strip().str.lower()
        df = df[["review", "sentiment"]].dropna()
        df["label"] = (df["sentiment"].str.lower() == "positive").astype(int)
        return df[["review", "label"]]

    # ── Strategy 2: aclImdb directory ────────────────────────────────────
    if os.path.isdir(acl_dir):
        print(f"[preprocess] Loading aclImdb directory → {acl_dir}")
        records = []
        for split in ("train", "test"):
            for sentiment, lbl in (("pos", 1), ("neg", 0)):
                folder = os.path.join(acl_dir, split, sentiment)
                if not os.path.isdir(folder):
                    continue
                for fname in os.listdir(folder):
                    fpath = os.path.join(folder, fname)
                    with open(fpath, encoding="utf-8", errors="ignore") as f:
                        records.append({"review": f.read(), "label": lbl})
        return pd.DataFrame(records)

    # ── Strategy 3: Demo dataset ──────────────────────────────────────────
    print("[preprocess] WARNING: No dataset found – generating a 200-row demo set.")
    print("             Place 'IMDB Dataset.csv' in the data/ folder for real results.")
    rng = np.random.default_rng(42)
    pos_phrases = [
        "brilliant film loved every moment",
        "outstanding performances and great story",
        "wonderful movie highly recommend",
        "amazing cinematography and acting",
        "masterpiece of modern cinema",
        "gripping story with excellent cast",
        "delightful and heart warming experience",
        "superb direction and stunning visuals",
    ]
    neg_phrases = [
        "terrible movie waste of time",
        "awful acting and boring plot",
        "dreadful film deeply disappointed",
        "horrible screenplay and bad direction",
        "worst movie ever seen",
        "dull and uninspiring story",
        "poorly written and badly acted",
        "complete disaster avoided at all costs",
    ]
    reviews, labels = [], []
    for _ in range(100):
        reviews.append(" ".join(rng.choice(pos_phrases, size=rng.integers(2, 5))))
        labels.append(1)
    for _ in range(100):
        reviews.append(" ".join(rng.choice(neg_phrases, size=rng.integers(2, 5))))
        labels.append(0)
    idx = rng.permutation(200)
    return pd.DataFrame({"review": np.array(reviews)[idx],
                          "label":  np.array(labels)[idx]})


# ── main pipeline ──────────────────────────────────────────────────────────
def run():
    ensure_nltk()
    stop_words = set(stopwords.words("english"))

    print("[preprocess] Loading dataset …")
    df = load_imdb_dataframe()
    print(f"[preprocess] Total samples: {len(df):,}  "
          f"(pos={df['label'].sum():,}, neg={(df['label']==0).sum():,})")

    print("[preprocess] Cleaning text …")
    df["clean"] = df["review"].astype(str).apply(
        lambda x: clean_text(x, stop_words)
    )

    # ── train / test split (80/20, stratified) ────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"],
        test_size=0.20, random_state=42, stratify=df["label"]
    )

    # ── TF-IDF vectorisation ──────────────────────────────────────────────
    print("[preprocess] Fitting TF-IDF vectoriser …")
    tfidf = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    # ── persist ───────────────────────────────────────────────────────────
    pd.DataFrame({"text": X_train.values, "label": y_train.values}).to_csv(
        TRAIN_CSV, index=False)
    pd.DataFrame({"text": X_test.values,  "label": y_test.values}).to_csv(
        TEST_CSV,  index=False)
    joblib.dump(tfidf, TFIDF_PATH)
    joblib.dump(X_train_tfidf, os.path.join(MODEL_DIR, "X_train_tfidf.joblib"))
    joblib.dump(X_test_tfidf,  os.path.join(MODEL_DIR, "X_test_tfidf.joblib"))
    joblib.dump(y_train.values, os.path.join(MODEL_DIR, "y_train.joblib"))
    joblib.dump(y_test.values,  os.path.join(MODEL_DIR, "y_test.joblib"))

    print(f"[preprocess] Train size : {X_train_tfidf.shape}")
    print(f"[preprocess] Test  size : {X_test_tfidf.shape}")
    print("[preprocess] Done – artefacts saved to model/")


if __name__ == "__main__":
    run()
