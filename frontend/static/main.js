/* main.js — frontend logic for IMDB Sentiment Analyzer */

const textarea    = document.getElementById("review-input");
const analyzeBtn  = document.getElementById("analyze-btn");
const clearBtn    = document.getElementById("clear-btn");
const charCount   = document.getElementById("char-count");
const resultCard  = document.getElementById("result-card");
const resultIcon  = document.getElementById("result-icon");
const resultLabel = document.getElementById("result-label");
const barNeg      = document.getElementById("bar-neg");
const barPos      = document.getElementById("bar-pos");
const pctNeg      = document.getElementById("pct-neg");
const pctPos      = document.getElementById("pct-pos");
const confText    = document.getElementById("confidence-text");
const errorBanner = document.getElementById("error-banner");
const errorText   = document.getElementById("error-text");
const exampleBtns = document.querySelectorAll(".example-chip");

// ── Character counter ───────────────────────────────────────────────────
textarea.addEventListener("input", () => {
  const n = textarea.value.length;
  charCount.textContent = `${n.toLocaleString()} character${n !== 1 ? "s" : ""}`;
});

// ── Clear ────────────────────────────────────────────────────────────────
clearBtn.addEventListener("click", () => {
  textarea.value = "";
  charCount.textContent = "0 characters";
  resultCard.classList.add("hidden");
  errorBanner.classList.add("hidden");
  textarea.focus();
});

// ── Example chips ────────────────────────────────────────────────────────
exampleBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    textarea.value = btn.dataset.text;
    const n = textarea.value.length;
    charCount.textContent = `${n.toLocaleString()} characters`;
    textarea.focus();
    // Auto-analyze
    analyzeBtn.click();
  });
});

// ── Analyze ──────────────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", async () => {
  const review = textarea.value.trim();

  if (!review) {
    showError("Please enter a movie review before analyzing.");
    return;
  }

  hideError();
  setLoading(true);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review }),
    });

    const data = await response.json();

    if (!response.ok) {
      showError(data.error || `Server error (${response.status})`);
      return;
    }

    showResult(data);
  } catch (err) {
    showError(
      "Could not reach the API. Make sure the Flask server is running on port 5000."
    );
    console.error(err);
  } finally {
    setLoading(false);
  }
});

// ── Also trigger on Ctrl/Cmd + Enter ─────────────────────────────────────
textarea.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    analyzeBtn.click();
  }
});

// ── Helpers ──────────────────────────────────────────────────────────────
function setLoading(on) {
  analyzeBtn.disabled = on;
  analyzeBtn.classList.toggle("loading", on);
  analyzeBtn.textContent = on ? "Analyzing…" : "Analyze Sentiment";
  if (on) analyzeBtn.classList.add("loading");
}

function showError(msg) {
  errorText.textContent = msg;
  errorBanner.classList.remove("hidden");
  resultCard.classList.add("hidden");
}

function hideError() {
  errorBanner.classList.add("hidden");
}

function showResult(data) {
  const isPos  = data.sentiment === "positive";
  const posP   = data.probabilities.positive;
  const negP   = data.probabilities.negative;

  // Icon + label
  resultIcon.textContent = isPos ? "😊" : "😞";
  resultLabel.textContent = isPos ? "Positive" : "Negative";
  resultLabel.className   = "result-label " + data.sentiment;

  // Confidence bar
  barPos.style.width = `${(posP * 100).toFixed(1)}%`;
  barNeg.style.width = `${(negP * 100).toFixed(1)}%`;
  pctPos.textContent = `${(posP * 100).toFixed(1)}%`;
  pctNeg.textContent = `${(negP * 100).toFixed(1)}%`;

  // Text
  const confPct = (data.confidence * 100).toFixed(1);
  confText.textContent =
    `Model confidence: ${confPct}% — The review is classified as ` +
    `${data.sentiment} with ${confPct}% certainty.`;

  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
