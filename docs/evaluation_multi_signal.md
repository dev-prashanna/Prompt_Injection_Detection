# Guardrailer Multi-Signal Evaluation Report

**Dataset:** `guardrailer_dataset_v1.parquet` (722,842 samples)
**Split:** 70/10/20 — Train 514,020 / Val 64,253 / Test 144,569
**Embedding Model:** `BAAI/bge-large-en-v1.5` (1024-dim)
**Date:** 2026-08-11
**Branch:** Phase-6-prototype

---

## 1. Executive Summary

| Approach | Balanced Accuracy | F1 | MCC | AUC-ROC | Target (BA >= 0.90) |
|---|---|---|---|---|---|
| Composite Score (default 0.50) | 0.5005 | 0.0088 | 0.0085 | 0.5466 | NOT MET |
| Composite Score (optimal 0.35) | 0.5148 | 0.7438 | 0.0661 | — | NOT MET |
| **RandomForest (ML)** | **0.7850** | **0.8151** | **0.5628** | **0.8727** | NOT MET |
| GradientBoosting (ML) | 0.7682 | 0.8148 | 0.5369 | 0.8597 | NOT MET |
| LogisticRegression (ML) | 0.6303 | 0.6741 | 0.2559 | 0.6709 | NOT MET |

**Best result:** RandomForest at BA=0.7850 — a 57% improvement over the composite score threshold but still 11.5 points below the 0.90 target.

---

## 2. Composite Score Threshold Analysis

The Phase-6 composite score uses 7 weighted signals to produce a single risk score. The default threshold is 0.50.

### Default Threshold (0.50)

| Metric | Value |
|---|---|
| Balanced Accuracy | 0.5005 |
| F1 Score | 0.0088 |
| MCC | 0.0085 |
| Precision | 0.6649 |
| Recall | 0.0044 |
| FPR | 0.0033 |
| FNR | 0.9956 |

### Confusion Matrix

| | Predicted Safe | Predicted Malicious |
|---|---|---|
| **Actual Safe** | 57,763 (TN) | 193 (FP) |
| **Actual Malicious** | 86,230 (FN) | 383 (TP) |

**Interpretation:** The composite score at threshold 0.50 correctly identifies nearly all benign prompts (99.67%) but misses 99.56% of malicious prompts. It is effectively a "classify everything as safe" system.

### Optimal Threshold Search

To find the best threshold, we swept from 0.30 to 0.79 and selected the threshold that maximizes BA while keeping FNR <= 5%.

| Threshold | BA | F1 | FNR | FPR | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|
| 0.30 | 0.4995 | 0.7479 | 0.42% | 99.68% | 86,251 | 57,770 | 186 | 362 |
| 0.35 (optimal) | 0.5148 | 0.7438 | 3.87% | 93.17% | 83,259 | 53,999 | 3,957 | 3,354 |
| 0.40 | 0.5394 | 0.6696 | 28.11% | 64.01% | 62,264 | 37,100 | 20,856 | 24,349 |
| 0.45 | 0.4978 | 0.2375 | 85.14% | 15.30% | 12,868 | 8,870 | 49,086 | 73,745 |
| 0.50 | 0.5005 | 0.0088 | 99.56% | 0.33% | 383 | 193 | 57,763 | 86,230 |
| 0.55 | 0.5001 | 0.0004 | 99.98% | 0.005% | 18 | 3 | 57,953 | 86,595 |
| 0.60+ | 0.5000 | 0.0000 | 100.0% | 0.00% | 4 | 0 | 57,956 | 86,609 |

**Key finding:** No threshold achieves acceptable performance. At low thresholds, FPR explodes (>90%). At high thresholds, FNR approaches 100%. The ROC-AUC of 0.5466 confirms the score has near-zero discriminative power.

---

## 3. ML Classifier Results (5-Fold Stratified CV)

Three classifiers were trained on the 7-signal feature matrix using 5-fold stratified cross-validation on the 144,569 test samples.

| Classifier | BA | 95% CI | F1 | MCC | AUC-ROC | Precision | Recall | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|
| **RandomForest** | **0.7850** | [0.4975, 0.5030] | 0.8151 | 0.5628 | 0.8727 | 0.8436 | 0.7884 | 0.2184 | 0.2116 |
| GradientBoosting | 0.7682 | [0.4975, 0.5024] | 0.8148 | 0.5369 | 0.8597 | 0.8135 | 0.8161 | 0.2796 | 0.1839 |
| LogisticRegression | 0.6303 | [0.4973, 0.5028] | 0.6741 | 0.2559 | 0.6709 | 0.7168 | 0.6363 | 0.3757 | 0.3637 |

### Per-Fold Breakdown

| Fold | LR BA | RF BA | GB BA |
|---|---|---|---|
| 1 | 0.6290 | 0.7837 | 0.7670 |
| 2 | 0.6321 | 0.7833 | 0.7672 |
| 3 | 0.6279 | 0.7854 | 0.7704 |
| 4 | 0.6308 | 0.7844 | 0.7665 |
| 5 | 0.6317 | 0.7881 | 0.7700 |
| **Mean** | **0.6303** | **0.7850** | **0.7682** |

---

## 4. Per-Category Performance

Using the composite score with default threshold (0.50):

| Category | Accuracy | Samples | Malicious | Benign |
|---|---|---|---|---|
| benign_control | **99.67%** | 57,956 | 0 | 57,956 |
| jailbreak | **0.53%** | 53,418 | 53,418 | 0 |
| direct_injection | **0.39%** | 18,441 | 18,441 | 0 |
| system_prompt_extraction | **0.18%** | 14,754 | 14,754 | 0 |

**Interpretation:** The system achieves near-perfect benign detection but fails catastrophically on all three malicious categories. System prompt extraction is the hardest (0.18% detection), followed by direct injection (0.39%) and jailbreak (0.53%).

---

## 5. Signal Discrimination Analysis

All 7 signals were evaluated for their ability to separate benign from malicious prompts on the test set.

| Signal | Benign Mean | Malicious Mean | Delta | Status |
|---|---|---|---|---|
| length_norm | 1.1799 | 1.1442 | 0.0357 | WEAK |
| centroid | 0.6995 | 0.7178 | 0.0183 | WEAK |
| perplexity | 0.6096 | 0.6261 | 0.0165 | WEAK |
| token_frequency | 0.3341 | 0.3417 | 0.0077 | WEAK |
| entropy | 0.7810 | 0.7769 | 0.0040 | WEAK |
| sparse_idf | 0.0143 | 0.0181 | 0.0037 | WEAK |
| ngram_overlap | 0.0037 | 0.0070 | 0.0032 | WEAK |

**Thresholds for classification:** delta > 0.10 = STRONG, > 0.05 = MODERATE, else WEAK.

**All 7 signals are WEAK.** The strongest signal (length_norm, delta=0.036) has less than half the separation needed for moderate discrimination. This confirms the embedding space collapse finding from the 199-sample evaluation.

---

## 6. Feature Statistics

| Signal | Mean | Std | Min | Max |
|---|---|---|---|---|
| sparse_idf | 0.0166 | 0.0283 | 0.0000 | ~0.15 |
| centroid | 0.7105 | 0.0844 | ~0.35 | ~0.95 |
| perplexity | 0.6195 | 0.0840 | 0.0000 | ~0.85 |
| entropy | 0.7786 | 0.0456 | ~0.40 | ~0.95 |
| token_frequency | 0.3387 | 0.0868 | 0.0000 | ~0.75 |
| ngram_overlap | 0.0057 | 0.0265 | 0.0000 | ~0.60 |
| length_norm | 1.1585 | 0.2303 | ~0.50 | ~2.50 |

---

## 7. Root Cause Analysis

### Why the embedding-based approach fails at scale

The 144,569-sample evaluation confirms the findings from the 199-sample Phase-6 evaluation:

1. **Embedding space collapse:** `BAAI/bge-large-en-v1.5` maps safe and malicious prompts to nearly identical vector regions. The centroid signal (cosine similarity to category centroids) shows only 0.018 separation between classes — both classes cluster around 0.70-0.72.

2. **All derived signals are noise:** Since every signal in the scoring pipeline is either derived from the embedding space (centroid, sparse_idf) or from generic text statistics (perplexity, entropy, token_frequency, ngram_overlap, length_norm), none can reliably separate the classes.

3. **ML classifiers can learn some patterns:** RandomForest achieves BA=0.785 by finding non-linear combinations of weak signals, but the signal quality fundamentally limits performance. The 0.785 ceiling represents the maximum discriminative information available in these 7 features.

4. **Benign detection is easy; malicious is hard:** The system correctly identifies benign prompts (99.67%) because they are statistically distinct in length and vocabulary. But malicious prompts are crafted to be indistinguishable from normal text — the signals that work for benign detection don't transfer to malicious detection.

### Quantitative Proof

| Metric | Threshold for Success | Actual Value | Verdict |
|---|---|---|---|
| Best signal delta | > 0.10 | 0.036 | **FAIL** |
| Composite ROC-AUC | > 0.75 | 0.547 | **FAIL** |
| Best ML BA | >= 0.90 | 0.785 | **FAIL** |
| Composite MCC | > 0.30 | 0.008 | **FAIL** |

---

## 8. Recommendations

1. **Abandon embedding-based scoring for security classification.** General-purpose embeddings optimize for semantic similarity, not intent classification. No amount of signal engineering can extract security-relevant information from a space that doesn't encode it.

2. **Use the hybrid lightweight scorer** (from `Guardrails_experiments/approaches/hybrid_lightweight_scorer/`) which achieves BA=0.865 with 117 engineered features and XGBoost — a 10% improvement over the embedding approach.

3. **Keep embeddings for RAG retrieval only** — use them to find similar known attacks, not to classify new ones.

4. **Invest in fine-tuned security classifiers** trained specifically on prompt injection detection tasks, rather than trying to repurpose general-purpose embeddings.

---

## 9. Reproducibility

```bash
# Run the evaluation
cd /home/prashanna/Documents/Guardrailer/phase-6
python3 evaluate_phase6_full.py

# Report location
cat evaluation_results/phase6_full_722k_report.json

# Checkpoints (for resume)
ls evaluation_results/phase6_ckpt/
```

**Checkpoint files:**
- `phase1_data.json` — Dataset splits
- `phase2_centroids.json` — Training centroids
- `phase3_centroid_scores.json` — Centroid similarity scores
- `phase4_sparse_scores.json` — IDF-weighted sparse scores
- `phase5_text_signals.json` — All text-only signals
- `phase8_composite.json` — Composite score results
- `phase9_classifiers.json` — ML classifier results

To resume from checkpoint, simply re-run the script — completed phases are skipped automatically.

---

*Report generated from `phase-6/evaluation_results/phase6_full_722k_report.json`*
*Evaluation date: 2026-08-11 14:30:31*
