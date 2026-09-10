"""Stage 2 of the distillation classifier: train on embeddings.

SCRATCH / STAGING (EVAL_LOG.md Round 18 -> distillation classifier). Takes the
LLM labels from quality_distill_label.py, embeds those chunks with the same
Cohere model the index uses, and trains a logistic-regression classifier.

Why embeddings: the Phase 2 re-index computes an embedding for every chunk
anyway, so at re-index time this classifier runs over all ~2.5M chunks at zero
marginal cost — no per-chunk API call, no second pass over the corpus.

Usage:
    python quality_distill_train.py
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from preproc import generate_text_embeddings, normalize_embeddings
from quality_filter_candidate import score_chunk_quality

LABELS_PATH = "quality_distill_labels.csv"
EMBED_CACHE = "quality_distill_embeddings.npy"
MODEL_PATH = "quality_distill_model.json"

EMBED_BATCH = 96  # Cohere embed v4 batch limit
SEED = 42
# "Bad" = anything the retrieval index is better off without.
POSITIVE_LABELS = {"GARBLED", "BOILERPLATE"}


def embed_texts(texts):
    """Embed with the same model/input_type the index uses for documents."""
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start:start + EMBED_BATCH]
        body = json.dumps(
            {"texts": batch, "input_type": "search_document"}, ensure_ascii=False
        ).encode("utf-8")
        resp = generate_text_embeddings(body)
        raw = resp.get("embeddings") or resp.get("embedding")
        vectors.extend(normalize_embeddings(raw, expected_n=len(batch)))
        print(f"  embedded {min(start + EMBED_BATCH, len(texts))}/{len(texts)}")
    return np.array(vectors, dtype=np.float32)


def main():
    df = pd.read_csv(LABELS_PATH)
    df = df[df["label"].isin({"COHERENT"} | POSITIVE_LABELS)].reset_index(drop=True)
    y = df["label"].isin(POSITIVE_LABELS).astype(int).values

    print(f"Labeled chunks: {len(df)}")
    print(df["label"].value_counts().to_string())
    print(f"Positive (drop-worthy) rate: {y.mean():.2%}\n")

    try:
        X = np.load(EMBED_CACHE)
        if len(X) != len(df):
            raise ValueError("cache size mismatch")
        print(f"Loaded cached embeddings {X.shape}")
    except (FileNotFoundError, ValueError):
        print("Embedding labeled chunks (one-time, ~$0.10)...")
        X = embed_texts(df["text"].astype(str).tolist())
        np.save(EMBED_CACHE, X)
        print(f"Cached embeddings to {EMBED_CACHE} {X.shape}")

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, np.arange(len(df)), test_size=0.25, random_state=SEED, stratify=y
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(X_tr, y_tr)

    pred = clf.predict(X_te)
    print("\n=== Embedding classifier (held-out 25%) ===")
    print(classification_report(y_te, pred, target_names=["keep", "drop"], digits=3))
    print("Confusion matrix [rows=true keep/drop, cols=pred keep/drop]:")
    print(confusion_matrix(y_te, pred))

    # Baseline: how much does the existing free heuristic already catch on the
    # same held-out set? Any claim that the classifier "helps" has to beat this.
    heur = np.array([
        1 if score_chunk_quality(t)["verdict"] == "drop" else 0
        for t in df.iloc[idx_te]["text"].astype(str)
    ])
    print("\n=== Existing free heuristic, same held-out set (baseline) ===")
    print(classification_report(y_te, heur, target_names=["keep", "drop"], digits=3))

    with open(MODEL_PATH, "w") as f:
        json.dump(
            {
                "coef": clf.coef_.tolist(),
                "intercept": clf.intercept_.tolist(),
                "n_train": int(len(y_tr)),
                "positive_labels": sorted(POSITIVE_LABELS),
            },
            f,
        )
    print(f"\nWrote {MODEL_PATH}")


if __name__ == "__main__":
    main()
