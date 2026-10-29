"""Train BECGuard.

Honest evaluation: the model is trained on the generated dataset and tested on the
ORIGINAL hand-written emails (data/holdout_original.csv), which are never used for
that evaluation run. The shipped model is then refit on everything.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline, make_union

BASE = Path(__file__).resolve().parents[1]


def build():
    words = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2),
                            max_df=.98, sublinear_tf=True)
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                            sublinear_tf=True, max_features=30000)
    return Pipeline([
        ("tfidf", make_union(words, chars)),
        ("clf", LogisticRegression(C=5, max_iter=3000, class_weight="balanced")),
    ])


def load(name):
    df = pd.read_csv(BASE / "data" / name)
    df["email_text"] = df["email_text"].fillna("").str.strip()
    return df[df.email_text != ""].drop_duplicates("email_text")


def main():
    train, hold = load("emails.csv"), load("holdout_original.csv")

    m = build().fit(train.email_text, train.label)
    pred = m.predict(hold.email_text)
    y = hold.label
    metrics = {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "test_size": len(y),
        "dataset_size": len(train) + len(hold),
        "evaluation": f"Trained on {len(train)} generated emails, tested on {len(y)} unseen hand-written emails",
    }

    full = pd.concat([train, hold])
    final = build().fit(full.email_text, full.label)
    (BASE / "models").mkdir(exist_ok=True)
    (BASE / "reports").mkdir(exist_ok=True)
    joblib.dump(final, BASE / "models" / "bec_model.joblib", compress=3)
    (BASE / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(classification_report(y, pred, digits=3, zero_division=0))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
