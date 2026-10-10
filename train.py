"""
train.py
────────
Trains a Logistic Regression classifier on the TF-IDF features produced by
preprocess.py and saves the fitted model to model/lr_model.joblib.
"""

import os
import joblib
from sklearn.linear_model import LogisticRegression

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


def run():
    print("[train] Loading TF-IDF features …")
    X_train = joblib.load(os.path.join(MODEL_DIR, "X_train_tfidf.joblib"))
    y_train = joblib.load(os.path.join(MODEL_DIR, "y_train.joblib"))

    print(f"[train] Training on {X_train.shape[0]:,} samples, "
          f"{X_train.shape[1]:,} features …")

    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    model_path = os.path.join(MODEL_DIR, "lr_model.joblib")
    joblib.dump(clf, model_path)
    print(f"[train] Model saved → {model_path}")


if __name__ == "__main__":
    run()
