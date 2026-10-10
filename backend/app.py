"""
backend/app.py
──────────────
Flask REST API that serves the trained sentiment model.

Endpoints
  GET  /            → health check JSON
  POST /predict     → { "review": "<text>" } → { "sentiment": "positive"|"negative",
                                                   "confidence": 0.93,
                                                   "probabilities": {"positive": 0.93, "negative": 0.07} }
  GET  /metrics     → last evaluation metrics (from reports/metrics.json)
"""

import os
import re
import json
import string
import joblib

import nltk
from nltk.corpus import stopwords
from flask import Flask, request, jsonify, send_from_directory

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(BASE_DIR)
MODEL_DIR   = os.path.join(ROOT_DIR, "model")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

# ── NLTK ───────────────────────────────────────────────────────────────────
for _res in ("stopwords", "punkt"):
    try:
        nltk.data.find(f"corpora/{_res}" if _res == "stopwords"
                       else f"tokenizers/{_res}")
    except LookupError:
        nltk.download(_res, quiet=True)

_STOP_WORDS = set(stopwords.words("english"))
_HTML_RE    = re.compile(r"<[^>]+>")
_SPACE_RE   = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.lower()
    text = _HTML_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = _SPACE_RE.sub(" ", text).strip().split()
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


# ── load model & vectoriser ────────────────────────────────────────────────
print("[api] Loading model artefacts …")
_tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
_clf   = joblib.load(os.path.join(MODEL_DIR, "lr_model.joblib"))
print("[api] Model ready.")

# ── Flask app ──────────────────────────────────────────────────────────────
app = Flask(__name__,
            static_folder=os.path.join(FRONTEND_DIR, "static"),
            template_folder=os.path.join(FRONTEND_DIR, "templates"))

app.config["JSON_SORT_KEYS"] = False


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/")
def health():
    return jsonify({"status": "ok", "message": "IMDB Sentiment API is running."})


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json(force=True, silent=True) or {}
    review = data.get("review", "").strip()

    if not review:
        return jsonify({"error": "Field 'review' is required and must be non-empty."}), 400

    cleaned   = clean_text(review)
    features  = _tfidf.transform([cleaned])
    proba     = _clf.predict_proba(features)[0]          # [neg_prob, pos_prob]
    label_idx = int(proba.argmax())
    sentiment = "positive" if label_idx == 1 else "negative"
    confidence = float(proba[label_idx])

    return jsonify({
        "sentiment":     sentiment,
        "confidence":    round(confidence, 4),
        "probabilities": {
            "positive": round(float(proba[1]), 4),
            "negative": round(float(proba[0]), 4),
        },
        "review_preview": review[:120] + ("…" if len(review) > 120 else ""),
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        return jsonify({"error": "Metrics not found. Run evaluate.py first."}), 404
    with open(metrics_path) as f:
        return jsonify(json.load(f))


@app.route("/report")
def report():
    report_path = os.path.join(REPORTS_DIR, "report_card.html")
    if not os.path.exists(report_path):
        return "<p>Report not generated yet. Run evaluate.py first.</p>", 404
    return send_from_directory(REPORTS_DIR, "report_card.html")


# ── serve frontend ─────────────────────────────────────────────────────────
@app.route("/app")
def frontend():
    return send_from_directory(
        os.path.join(FRONTEND_DIR, "templates"), "index.html"
    )


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(os.path.join(FRONTEND_DIR, "static"), filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
