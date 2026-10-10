# IMDB Sentiment Analysis — Project Documentation

## Overview

A complete end-to-end sentiment analysis pipeline trained on the 50,000-review
IMDB dataset.  The project covers text preprocessing, TF-IDF feature extraction,
Logistic Regression classification, model evaluation, a REST API, and a web UI.

---

## Project Structure

```
imdb-sentiment/
├── data/                          # Dataset files (place IMDB Dataset.csv here)
├── model/                         # Saved model artefacts (auto-generated)
├── reports/                       # Evaluation outputs (auto-generated)
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── sentiment_distribution.png
│   └── report_card.html
├── frontend/
│   ├── templates/index.html       # Web UI
│   └── static/
│       ├── style.css
│       └── main.js
├── backend/
│   └── app.py                     # Flask REST API
├── preprocess.py                  # Data cleaning + TF-IDF
├── train.py                       # Model training
├── evaluate.py                    # Metrics + charts + report card
├── run_pipeline.py                # One-command runner
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Obtain the dataset

Download **IMDB Dataset.csv** from
[Kaggle – IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
and place it in the `data/` folder.

> **No dataset?** The pipeline auto-generates a small 200-row demo set so you
> can verify the full flow end-to-end before adding the real data.

### 3. Run the full pipeline

```bash
python run_pipeline.py
```

This executes four steps in order:

| Step | Script | Output |
|------|--------|--------|
| 1 Preprocessing | `preprocess.py` | `model/tfidf_vectorizer.joblib`, split CSVs |
| 2 Training       | `train.py`       | `model/lr_model.joblib` |
| 3 Evaluation     | `evaluate.py`    | `reports/` (JSON, PNG, HTML) |
| 4 API server     | `backend/app.py` | http://localhost:5000 |

Skip training if the model already exists:

```bash
python run_pipeline.py --skip-train
```

Run pipeline only (no server):

```bash
python run_pipeline.py --no-server
```

---

## Preprocessing Pipeline

| Step | Detail |
|------|--------|
| Lowercase | `text.lower()` |
| HTML stripping | regex `<[^>]+>` → space |
| Punctuation removal | `str.maketrans` with `string.punctuation` |
| Digit removal | regex `\d+` → space |
| Tokenisation | whitespace split |
| Stop-word removal | NLTK English stop-word list |
| Token filter | discard tokens of length ≤ 1 |

---

## TF-IDF Vectorisation

| Parameter | Value |
|-----------|-------|
| `max_features` | 50,000 |
| `ngram_range` | (1, 2) — unigrams + bigrams |
| `min_df` | 2 |
| `sublinear_tf` | True |

---

## Model — Logistic Regression

| Parameter | Value |
|-----------|-------|
| Solver | lbfgs |
| Regularisation C | 1.0 |
| Max iterations | 1,000 |
| Random state | 42 |
| Train / Test split | 80 % / 20 % (stratified) |

Typical performance on the full 50 K IMDB dataset:

| Metric | Score |
|--------|-------|
| Accuracy | ~89 % |
| Precision | ~89 % |
| Recall | ~89 % |
| F1-Score | ~89 % |

---

## API Reference

### `POST /predict`

**Request body (JSON)**

```json
{ "review": "An absolute masterpiece. The acting was phenomenal." }
```

**Response**

```json
{
  "sentiment": "positive",
  "confidence": 0.9743,
  "probabilities": {
    "positive": 0.9743,
    "negative": 0.0257
  },
  "review_preview": "An absolute masterpiece. The acting was phenomenal."
}
```

### `GET /metrics`

Returns the evaluation metrics from `reports/metrics.json`.

### `GET /report`

Serves the self-contained HTML report card.

### `GET /app`

Serves the web UI.

---

## Frontend

Open **http://localhost:5000/app** in your browser.

- Paste any movie review into the text box and click **Analyze Sentiment**.
- The bar shows the split of positive / negative probability.
- Four example reviews are provided for quick testing.
- `Ctrl+Enter` / `Cmd+Enter` triggers analysis from the keyboard.

---

## Report Card

`reports/report_card.html` is a self-contained HTML file (no external assets)
that includes:

- Key metric cards (accuracy, precision, recall, F1)
- Per-class classification report table
- Confusion matrix heatmap (base64-embedded PNG)
- Sentiment distribution bar chart (base64-embedded PNG)
- Model and pipeline details

Open it directly in any browser or serve it via `GET /report`.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| scikit-learn | TF-IDF, Logistic Regression, metrics |
| pandas | DataFrame handling |
| numpy | Numerical operations |
| nltk | Stop-words, tokenisation |
| matplotlib | Chart generation |
| seaborn | Confusion matrix heatmap |
| flask | REST API server |
| joblib | Model serialisation |
