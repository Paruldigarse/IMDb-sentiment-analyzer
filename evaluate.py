"""
evaluate.py
───────────
Loads the trained Logistic Regression model, evaluates it on the held-out
test set, and writes:
  • reports/metrics.json          – accuracy / precision / recall / F1
  • reports/confusion_matrix.png  – heatmap
  • reports/sentiment_distribution.png – bar chart
  • reports/report_card.html      – self-contained HTML report card
"""

import os
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
MODEL_DIR   = os.path.join(BASE_DIR, "model")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

LABELS = ["Negative", "Positive"]
COLORS = {"Negative": "#e05252", "Positive": "#4caf7d"}


# ── helpers ─────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, path: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS, ax=ax,
        linewidths=0.5, linecolor="#dddddd",
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title("Confusion Matrix",  fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Confusion matrix → {path}")


def plot_sentiment_distribution(y_true: np.ndarray, y_pred: np.ndarray, path: str):
    true_pos  = int(y_true.sum())
    true_neg  = int(len(y_true) - true_pos)
    pred_pos  = int(y_pred.sum())
    pred_neg  = int(len(y_pred) - pred_pos)

    x      = np.arange(2)
    width  = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    bars1 = ax.bar(x - width/2, [true_neg,  true_pos],  width,
                   label="Actual",    color=["#e05252", "#4caf7d"], alpha=0.85)
    bars2 = ax.bar(x + width/2, [pred_neg,  pred_pos],  width,
                   label="Predicted", color=["#c0392b", "#27ae60"], alpha=0.85)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 20,
                f"{h:,}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Sentiment Distribution – Actual vs Predicted",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Sentiment distribution → {path}")


def encode_image_b64(path: str) -> str:
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def generate_report_card(metrics: dict, cm_path: str, dist_path: str, out_path: str):
    cm_b64   = encode_image_b64(cm_path)
    dist_b64 = encode_image_b64(dist_path)

    def pct(v):
        return f"{v * 100:.2f}%"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IMDB Sentiment – Evaluation Report</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    background: #f7f8fa; color: #1f2328; font-size: 14px; line-height: 1.6;
    padding: 32px 16px 48px;
  }}
  .page {{ max-width: 820px; margin: 0 auto; }}
  h1 {{ font-size: 22px; font-weight: 700; margin-bottom: 4px; }}
  .subtitle {{ color: #57606a; font-size: 13px; margin-bottom: 28px; }}
  h2 {{ font-size: 15px; font-weight: 600; margin-bottom: 12px;
        border-left: 3px solid #3b82d4; padding-left: 8px; color: #1f2328; }}
  .card {{
    background: #fff; border: 1px solid #e5e7eb;
    border-radius: 8px; padding: 20px 24px; margin-bottom: 20px;
  }}
  .metrics-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
  }}
  .metric-box {{
    background: #f7f8fa; border: 1px solid #e5e7eb;
    border-radius: 6px; padding: 14px 10px; text-align: center;
  }}
  .metric-box .value {{
    font-size: 26px; font-weight: 700; color: #3b82d4;
  }}
  .metric-box .label {{
    font-size: 11px; color: #57606a; text-transform: uppercase;
    letter-spacing: 0.05em; margin-top: 4px;
  }}
  .charts-row {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
  }}
  .chart-box {{ text-align: center; }}
  .chart-box img {{ width: 100%; border-radius: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 8px 12px; text-align: left;
             border-bottom: 1px solid #e5e7eb; }}
  th {{ background: #f7f8fa; font-weight: 600; color: #57606a;
        text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; }}
  tr:last-child td {{ border-bottom: none; }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 11px; font-weight: 600;
  }}
  .pos {{ background: #d4edda; color: #155724; }}
  .neg {{ background: #f8d7da; color: #721c24; }}
  .footer {{
    text-align: center; font-size: 11px; color: #8b949e;
    border-top: 1px solid #e5e7eb; padding-top: 16px; margin-top: 8px;
  }}
  @media (max-width: 600px) {{
    .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .charts-row   {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="page">

  <h1>IMDB Sentiment Analysis — Evaluation Report</h1>
  <p class="subtitle">Logistic Regression · TF-IDF (unigrams + bigrams, 50 K features) · 80/20 train-test split</p>

  <!-- ── Key Metrics ───────────────────────────────────────────────────── -->
  <div class="card">
    <h2>Key Metrics</h2>
    <div class="metrics-grid">
      <div class="metric-box">
        <div class="value">{pct(metrics['accuracy'])}</div>
        <div class="label">Accuracy</div>
      </div>
      <div class="metric-box">
        <div class="value">{pct(metrics['precision'])}</div>
        <div class="label">Precision</div>
      </div>
      <div class="metric-box">
        <div class="value">{pct(metrics['recall'])}</div>
        <div class="label">Recall</div>
      </div>
      <div class="metric-box">
        <div class="value">{pct(metrics['f1'])}</div>
        <div class="label">F1-Score</div>
      </div>
    </div>
  </div>

  <!-- ── Per-class Report ──────────────────────────────────────────────── -->
  <div class="card">
    <h2>Per-class Classification Report</h2>
    <table>
      <thead>
        <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><span class="badge neg">Negative</span></td>
          <td>{pct(metrics['report']['0']['precision'])}</td>
          <td>{pct(metrics['report']['0']['recall'])}</td>
          <td>{pct(metrics['report']['0']['f1-score'])}</td>
          <td>{int(metrics['report']['0']['support']):,}</td>
        </tr>
        <tr>
          <td><span class="badge pos">Positive</span></td>
          <td>{pct(metrics['report']['1']['precision'])}</td>
          <td>{pct(metrics['report']['1']['recall'])}</td>
          <td>{pct(metrics['report']['1']['f1-score'])}</td>
          <td>{int(metrics['report']['1']['support']):,}</td>
        </tr>
      </tbody>
    </table>
  </div>

  <!-- ── Confusion Matrix ──────────────────────────────────────────────── -->
  <div class="card">
    <h2>Confusion Matrix &amp; Sentiment Distribution</h2>
    <div class="charts-row">
      <div class="chart-box">
        <img src="data:image/png;base64,{cm_b64}" alt="Confusion Matrix">
      </div>
      <div class="chart-box">
        <img src="data:image/png;base64,{dist_b64}" alt="Sentiment Distribution">
      </div>
    </div>
  </div>

  <!-- ── Model Details ────────────────────────────────────────────────────-->
  <div class="card">
    <h2>Model &amp; Pipeline Details</h2>
    <table>
      <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>Classifier</td><td>Logistic Regression (lbfgs solver)</td></tr>
        <tr><td>Regularisation C</td><td>1.0</td></tr>
        <tr><td>Vectoriser</td><td>TF-IDF · max_features=50,000 · ngram_range=(1,2)</td></tr>
        <tr><td>Sublinear TF scaling</td><td>Yes</td></tr>
        <tr><td>Train / Test split</td><td>80 % / 20 % (stratified)</td></tr>
        <tr><td>Stop-word removal</td><td>NLTK English stop-words</td></tr>
        <tr><td>HTML stripping</td><td>Yes (regex)</td></tr>
        <tr><td>Punctuation removal</td><td>Yes</td></tr>
        <tr><td>Test samples</td><td>{int(metrics['report']['0']['support']) + int(metrics['report']['1']['support']):,}</td></tr>
      </tbody>
    </table>
  </div>

  <div class="footer">Made with IBM Bob</div>
</div>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[evaluate] Report card → {out_path}")


# ── main ────────────────────────────────────────────────────────────────────
def run():
    print("[evaluate] Loading model and test data …")
    clf      = joblib.load(os.path.join(MODEL_DIR, "lr_model.joblib"))
    X_test   = joblib.load(os.path.join(MODEL_DIR, "X_test_tfidf.joblib"))
    y_test   = joblib.load(os.path.join(MODEL_DIR, "y_test.joblib"))

    print("[evaluate] Running predictions …")
    y_pred = clf.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)
    rep  = classification_report(y_test, y_pred, output_dict=True)

    print("\n" + "─" * 50)
    print(f"  Accuracy  : {acc  * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec  * 100:.2f}%")
    print(f"  F1-Score  : {f1   * 100:.2f}%")
    print("─" * 50 + "\n")
    print(classification_report(y_test, y_pred,
                                 target_names=["Negative", "Positive"]))

    metrics = {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "confusion_matrix": cm.tolist(),
        "report":    {str(k): v for k, v in rep.items()
                      if k in ("0", "1", "macro avg", "weighted avg")},
    }

    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_path, "w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"[evaluate] Metrics → {metrics_path}")

    cm_path   = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    dist_path = os.path.join(REPORTS_DIR, "sentiment_distribution.png")

    plot_confusion_matrix(cm, cm_path)
    plot_sentiment_distribution(y_test, y_pred, dist_path)
    generate_report_card(metrics, cm_path, dist_path,
                         os.path.join(REPORTS_DIR, "report_card.html"))

    return metrics


if __name__ == "__main__":
    run()
