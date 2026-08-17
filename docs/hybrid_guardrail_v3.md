# Guardrailer v3 — Full Results Interpretation & Analysis

**Source:** `notebook1432444077.ipynb`
**Date:** August 2026
**Hardware:** Kaggle P100-PCIE-16GB | **Pipeline:** Regex Pattern Scorer + Quantized Similarity + Stacking Ensemble
**Adaptation:** Online threshold + weight tuning | **Inference:** ~0.149ms/prompt (~6,727/sec)

---

## Table of Contents

1. [Environment & Configuration](#1-environment--configuration)
2. [Mathematical Framework](#2-mathematical-framework)
3. [Confusion Matrix Analysis](#3-confusion-matrix-analysis)
4. [ROC Curve Analysis](#4-roc-curve-analysis)
5. [Precision-Recall Curve Analysis](#5-precision-recall-curve-analysis)
6. [Score Distribution Analysis](#6-score-distribution-analysis)
7. [Adversarial Robustness Analysis](#7-adversarial-robustness-analysis)
8. [Feature Importance Analysis](#8-feature-importance-analysis)
9. [Cross-Dataset Generalization Analysis](#9-cross-dataset-generalization-analysis)
10. [Dashboard Summary](#10-dashboard-summary)
11. [Pros & Cons](#11-pros--cons)
12. [How to Upgrade](#12-how-to-upgrade)
13. [Building a More Robust Model](#13-building-a-more-robust-model)
14. [Issues & Fixes](#14-issues--fixes)

---

## 1. Environment & Configuration

| Parameter | Value |
|---|---|
| GPU | Tesla P100-PCIE-16GB |
| CPU cores | 4 |
| RAM | 33.7 GB (avail: 31.9 GB) |
| Seed | 42 |
| Dataset rows | 722,842 |
| Train / Val / Test | 578,273 / 72,284 / 72,285 |
| TF-IDF word max features | 5,000 |
| TF-IDF n-gram range | (1, 2) |
| Similarity word max | 20,000 |
| Similarity char max | 10,000 |
| XGBoost/LGB n_estimators | 500 |
| XGBoost/LGB max_depth | 6 |
| Learning rate | 0.05 |
| Pattern gate threshold | 0.6 |
| Ensemble threshold | 0.55 (calibrated to 0.44) |
| Ensemble weights (initial) | [0.10, 0.10, 0.80] |
| RF n_estimators | 300 |
| RF max_depth | 12 |
| Stacking cv | 3 |
| Final estimator | LogisticRegression(C=1.0, max_iter=1000) |
| Calibration method | Isotonic regression (cv=3) |
| Checkpoint interval | 300s |
| Total runtime | ~5 hours |

---

## 2. Mathematical Framework

### 2.1 Feature Engineering (117 Features)

The model extracts 117 handcrafted features from each prompt. Key mathematical formulations:

#### Shannon Entropy (Character-level)

$$H_{char} = -\sum_{c \in \text{freq}} \frac{c}{N} \log_2\left(\frac{c}{N}\right)$$

Where $N$ is the total character count and $c$ is the frequency of each unique character. Measures randomness in character distribution — higher entropy indicates more diverse character usage.

#### Shannon Entropy (Word-level)

$$H_{word} = -\sum_{w \in \text{wf}} \frac{w}{N_w} \log_2\left(\frac{w}{N_w}\right)$$

Where $N_w$ is the total word count and $w$ is the frequency of each unique word.

#### N-gram Entropy (Normalized)

$$H_{ngram}(t, n) = \frac{-\sum_{g \in \text{freq}} \frac{g}{T} \log_2\left(\frac{g}{T}\right)}{\log_2(|\text{freq}|)}$$

Where $T$ is the total number of n-grams, $g$ is the frequency of each unique n-gram, and $|\text{freq}|$ is the number of unique n-grams. Normalized by maximum possible entropy ($\log_2(|\text{freq}|)$) to produce a value in [0, 1].

#### Flesch-Kincaid Grade Level

$$FK = 0.39 \times \frac{W}{S} + 11.8 \times \frac{SYL}{W} - 15.59$$

Where $W$ = word count, $S$ = sentence count, $SYL$ = total syllable count. Measures readability complexity.

#### Coleman-Liau Index

$$CL = 0.0588 \times \frac{L}{W} \times 100 - 0.296 \times \frac{S}{W} \times 100 - 15.8$$

Where $L$ = letter count, $W$ = word count, $S$ = sentence count.

#### Syllable Count (Simplified)

```python
def _sc(w):
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for c in w:
        is_vowel = c in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if w.endswith("e"): count -= 1
    return max(1, count)
```

Counts vowel groups (consecutive vowels count as one syllable), with special handling for trailing 'e'.

#### Sliding Window TTR (Type-Token Ratio)

$$TTR_{slide}(w) = \frac{1}{|\text{windows}|} \sum_{i=0}^{n-w} \frac{|\text{unique}(ws[i:i+w])|}{w}$$

Measures lexical diversity over sliding windows of size $w$ (50 or 100 words). Uses step = $w/2$ for efficiency.

#### Sentence Length Variance

$$SV = \text{Var}(|s_1|, |s_2|, \ldots, |s_n|)$$

Where $|s_i|$ is the word count of sentence $i$. High variance indicates mixed short/long sentences.

#### Markdown Density

$$MD = \min\left(1.0, \frac{N_{md} + N_{link} + N_{bullet} + N_{num}}{|t|}\right)$$

#### Escape Character Density

$$ESC = \min\left(1.0, \frac{|\text{matches of } \\[nrtbfav\\'"0]|}{|W|}\right)$$

#### Nesting Depth (Brackets)

Tracks maximum depth of unmatched `(`, `[`, `{` characters. Higher depth indicates structured/injected content.

#### Bigram/Trigram Repeat Ratio

$$Repeat_n = 1 - \frac{|\text{unique n-grams}|}{|\text{total n-grams}|}$$

High repeat ratio indicates repetitive text (common in jailbreak attempts).

#### Bigram/Trigram Diversity

$$Diversity_n = \frac{|\text{unique n-grams}|}{|\text{total n-grams}|}$$

Inverse of repeat ratio. Low diversity indicates template-like text.

#### Word Length Entropy

$$H_{wl} = H_{ngram}(\text{join}(|w_1|, |w_2|, \ldots, |w_n|), 1)$$

Entropy of the sequence of word lengths (as characters).

### 2.2 TF-IDF Similarity (Quantized)

#### Quantized Cosine Similarity

$$sim_{quant}(q, c) = \frac{1}{1 + e^{-5.0 \times \left(\frac{q \cdot c}{127} - 0.3\right)}}$$

Where:
- $q$ = quantized query vector (int8, values in [-127, 127])
- $c$ = quantized centroid vector (int8)
- The dot product is divided by 127 to normalize to [-1, 1]
- Sigmoid with steepness=5.0 and shift=0.3 maps to [0, 1]

The quantization: $q = \text{round}(d_{norm} \times 127)$ where $d_{norm} = d / \|d\|_2$

This is a **memory-efficient approximation** — int8 vectors use 4x less memory than float32, enabling the 20K word + 10K char vocabulary to fit in GPU memory.

#### IDF Keyword Score

$$score_{kw}(t) = \frac{1}{1 + e^{-10.0 \times \left(\frac{\sum_{w \in t} IDF(w) \times \frac{DF_{mal}(w)}{DF_{mal}(w) + DF_{safe}(w) + 1}}{|t|} - 0.1\right)}}$$

Where $IDF(w) = \log\left(\frac{N - DF(w) + 0.5}{DF(w) + 0.5} + 1.0\right)$

### 2.3 Regex Pattern Scorer

47 regex patterns with associated confidence scores (0.6–0.95). The scorer returns the maximum matching score:

$$score_{regex}(t) = \max_{p \in \text{patterns}} \begin{cases} s_p & \text{if } p \text{ matches } t \\ 0 & \text{otherwise} \end{cases}$$

### 2.4 Stacking Ensemble

The stacking classifier combines 3 base learners with a meta-learner:

**Base learners:**
1. **XGBoost** — 500 trees, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8
2. **LightGBM** — same hyperparameters
3. **RandomForest** — 300 trees, max_depth=12, min_samples_split=5

**Meta-learner:** LogisticRegression(C=1.0, max_iter=1000)

**Stacking formula:**

$$P(y=1|x) = \sigma\left(\beta_0 + \beta_1 \cdot P_{xgb}(x) + \beta_2 \cdot P_{lgb}(x) + \beta_3 \cdot P_{rf}(x)\right)$$

Where $\sigma$ is the logistic function and $P_{model}(x)$ are the predict_proba outputs from each base learner. With `passthrough=True`, the original features are also passed to the meta-learner:

$$P(y=1|x) = \sigma\left(\beta_0 + \beta_1 \cdot P_{xgb}(x) + \beta_2 \cdot P_{lgb}(x) + \beta_3 \cdot P_{rf}(x) + \beta_4 \cdot x\right)$$

### 2.5 Isotonic Calibration

The stacking classifier's output probabilities are calibrated using isotonic regression:

$$\hat{p}_{cal} = f_{isotonic}(\hat{p}_{stk})$$

Where $f_{isotonic}$ is a non-parametric monotonic function fit on the calibration set. This transforms the raw stacking probabilities into well-calibrated probabilities where $\hat{p}_{cal} \approx P(y=1|\hat{p}_{cal})$.

### 2.6 Ensemble Decision (Final Score)

$$ensemble = w_0 \cdot score_{regex} + w_1 \cdot score_{sim} + w_2 \cdot P_{cal}(x)$$

Where:
- $w_0, w_1, w_2$ = adaptive weights (initial: [0.10, 0.10, 0.80])
- $score_{regex}$ = regex pattern score (gated: $\geq$ 0.6 threshold)
- $score_{sim}$ = TF-IDF similarity score
- $P_{cal}(x)$ = calibrated stacking probability

**Decision:** $pred = \begin{cases} 1 & \text{if } ensemble \geq threshold \\ 0 & \text{otherwise} \end{cases}$

### 2.7 Adaptive Threshold (EMA)

$$fp_{ema}^{(t)} = \alpha \cdot fp^{(t)} + (1 - \alpha) \cdot fp_{ema}^{(t-1)}$$

$$threshold^{(t)} = threshold^{(t-1)} - \alpha \cdot (fp_{ema}^{(t)} - target_{fp})$$

$$threshold^{(t)} = \max(0.30, \min(0.80, threshold^{(t)}))$$

Where $\alpha = 0.001$ and $target_{fp} = 0.05$. The threshold adapts to maintain a target false positive rate of 5%.

### 2.8 Adaptive Weights (Softmax)

$$acc^{(t)} = \alpha \cdot acc_{comp}^{(t)} + (1 - \alpha) \cdot acc^{(t-1)}$$

$$w_i^{(t)} = \frac{e^{acc_i^{(t)}}}{\sum_j e^{acc_j^{(t)}}}$$

Where $\alpha = 0.05$ and $acc_{comp} = [1 - fp_{rate}, 1 - fn_{rate}, accuracy]$. The weights are updated via softmax to emphasize the best-performing component.

### 2.9 Expected Calibration Error (ECE)

$$ECE = \sum_{i=1}^{B} \frac{|B_i|}{N} \left| \text{acc}(B_i) - \text{conf}(B_i) \right|$$

Where $B$ = number of bins (10), $|B_i|$ = number of samples in bin $i$, $N$ = total samples, $\text{acc}(B_i)$ = mean true label in bin, $\text{conf}(B_i)$ = mean predicted probability in bin. Lower ECE indicates better calibration.

### 2.10 Matthews Correlation Coefficient (MCC)

$$MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

Ranges from -1 to +1. +1 = perfect prediction, 0 = random, -1 = total disagreement. More robust than accuracy for imbalanced datasets.

### 2.11 Cohen's Kappa

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

Where $p_o$ = observed accuracy, $p_e$ = expected accuracy by chance. Measures inter-rater agreement beyond random chance.

### 2.12 Balanced Accuracy

$$BA = \frac{1}{2}\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right)$$

### 2.13 Drift Detection

$$\text{drift} = \begin{cases} 2 & \text{if } |\mu_{recent} - \mu_{baseline}| > 2 \cdot threshold \\ 1 & \text{if } |\mu_{recent} - \mu_{baseline}| > threshold \\ 0 & \text{otherwise} \end{cases}$$

Where $\mu_{recent}$ = mean of last 200 predictions, $\mu_{baseline}$ = baseline mean. Recalibration triggered when drift_count $\geq$ 3.

---

## 3. Confusion Matrix Analysis

**File:** `confusion_matrix.png`

|  | Predicted Safe | Predicted Malicious |
|---|---|---|
| **True Safe** | **21,968** (TN) | **7,010** (FP) |
| **True Malicious** | **4,707** (FN) | **38,600** (TP) |

### Key Observations

- **True Positives (38,600):** The model correctly identifies 89.1% of malicious prompts. The dark blue cell dominates the matrix, indicating strong attack detection.
- **True Negatives (21,968):** Only 75.8% of safe prompts are correctly classified. The lighter blue shade relative to TP shows the model is weaker at recognizing safe content.
- **False Positives (7,010):** This is the **critical weakness**. 24.2% of safe prompts are incorrectly flagged as malicious. In production, this means 1 in 4 legitimate user queries would be blocked.
- **False Negatives (4,707):** 10.9% of attacks slip through. These are dangerous — they represent successful jailbreaks.

### Impact Assessment

The model is **biased toward high recall** (catching attacks) at the cost of **low precision on safe class** (blocking legitimate users). For a production guardrail:
- **FP cost:** User frustration, lost functionality, reduced trust
- **FN cost:** Security breach, harmful content generation
- The current balance favors security over usability, which may be acceptable for high-risk applications but problematic for general-purpose chatbots.

---

## 4. ROC Curve Analysis

**File:** `roc_curve.png`

| Metric | Value |
|---|---|
| **AUC-ROC** | 0.9082 |
| **Shape** | Steep initial rise, smooth plateau |

### Interpretation

The ROC curve shows the trade-off between True Positive Rate (sensitivity) and False Positive Rate (1 - specificity) across all threshold values.

- **AUC = 0.9082** indicates excellent discrimination ability. The model ranks malicious prompts higher than safe prompts 90.82% of the time.
- **At FPR = 0.10:** TPR $\approx$ 0.82 — catching 82% of attacks while only misclassifying 10% of safe prompts.
- **At FPR = 0.20:** TPR $\approx$ 0.90 — catching 90% of attacks at 20% FP rate.
- **The curve's steep initial slope** means the model's highest-confidence predictions are very accurate.
- **The plateau at high FPR** indicates diminishing returns — going from 80% to 90% recall costs significant FP increase.

### Mathematical Insight

The AUC is equivalent to the probability that a randomly chosen malicious prompt scores higher than a randomly chosen safe prompt:

$$AUC = P(S_{mal} > S_{safe})$$

The value 0.9082 means there's a 9.18% chance of a safe prompt outranking a malicious one.

---

## 5. Precision-Recall Curve Analysis

**File:** `pr_curve.png`

| Metric | Value |
|---|---|
| **Average Precision (AP)** | 0.9353 |
| **Precision at Recall=1.0** | $\approx$ 0.60 |
| **Precision at Recall=0.8** | $\approx$ 0.95 |

### Interpretation

The PR curve is particularly informative for imbalanced datasets (here ~60/40 split).

- **AP = 0.9353** is excellent — the model maintains high precision across most recall levels.
- **At Recall = 0.8:** Precision $\approx$ 0.95, meaning 95% of detected attacks are real attacks.
- **At Recall = 0.9:** Precision $\approx$ 0.88, still strong.
- **At Recall = 1.0:** Precision drops to $\approx$ 0.60, reflecting the 7,010 false positives needed to catch all attacks.
- **The sharp initial spike** (precision = 1.0 at very low recall) confirms the model's highest-confidence predictions are perfect.

### Why PR > ROC for This Task

With 60% malicious and 40% safe samples, ROC can be misleadingly optimistic. PR curve better reflects the actual cost of false positives in the context of this class distribution.

---

## 6. Score Distribution Analysis

**File:** `score_distribution.png`

| Feature | Observation |
|---|---|
| **Safe distribution** | Concentrated at 0.05–0.25, peak at $\approx$ 0.15 |
| **Malicious distribution** | Concentrated at 0.85–1.0, peak at $\approx$ 0.98 |
| **Threshold** | 0.440 (green dashed line) |
| **Overlap region** | 0.30–0.60 |

### Interpretation

The bimodal separation is strong but imperfect:

- **Safe prompts** cluster tightly around low scores (mean $\approx$ 0.15). The distribution has a long right tail extending to 0.60 — these are the "borderline safe" prompts that get false-positive'd.
- **Malicious prompts** cluster tightly around high scores (mean $\approx$ 0.95). The sharp peak at 1.0 shows many attacks are caught with high confidence.
- **The threshold at 0.44** is positioned in the overlap zone. Lowering it would catch more attacks but increase FPs; raising it would reduce FPs but miss more attacks.
- **The overlap zone (0.30–0.60)** is where the model is uncertain. This represents $\approx$ 10–15% of prompts.

### Mathematical Insight

The optimal threshold minimizes:

$$J(t) = TPR(t) - FPR(t)$$

The calibrated threshold (0.44) is likely near the Youden's J optimum, trading off equal amounts of TPR and FPR.

---

## 7. Adversarial Robustness Analysis

**File:** `robustness.png`

| Transform | Detection Rate | Status | Time |
|---|---|---|---|
| Original | 92.5% | PASS | 1.2s |
| Paraphrase | 91.5% | PASS | 1.3s |
| **Misspelling** | **63.5%** | **WARN** | 1.3s |
| Unicode | 83.5% | PASS | 1.3s |
| Base64 | 100.0% | PASS | 1.3s |
| Roleplay | 96.5% | PASS | 1.3s |
| Euphemistic | 92.5% | PASS | 1.3s |
| Long context | 93.5% | PASS | 1.5s |
| **Average** | **88.7%** | | |

### Detailed Analysis by Transform

#### Original (92.5%)
The baseline detection rate on unmodified malicious prompts. 7.5% of attacks evade detection even without obfuscation — these are likely sophisticated prompts that avoid common keyword/pattern triggers.

#### Paraphrase (91.5%)
Word substitution using synonyms (e.g., "ignore" → "disregard", "previous" → "prior"). The model handles this well because:
- TF-IDF char n-grams capture subword patterns
- Structural features (imperative verbs, second-person ratio) are preserved
- Only 1% drop from original shows semantic robustness

#### Misspelling (63.5%) — CRITICAL WEAKNESS
Character substitution (e.g., "ignore" → "ign0re", "previous" → "prev1ous"). The 29% drop from original is caused by:
- TF-IDF features are character-exact — "ign0re" ≠ "ignore" in the vocabulary
- Regex patterns are character-exact — they won't match obfuscated versions
- The handcrafted features (keyword counts, pattern density) all depend on exact string matching

#### Unicode (83.5%)
Homoglyph substitution (e.g., Latin 'a' → Cyrillic 'а'). The 9% drop indicates partial resilience through:
- Character entropy features detecting unusual Unicode distributions
- `unicode_anomaly_score` feature flagging homoglyphs
- But many Unicode variants still evade detection

#### Base64 (100.0%)
Encoded payloads with "Decode and follow:" prefix. Perfect detection because:
- `has_encoded_payload_hint` feature directly matches the prefix pattern
- The encoding instruction ratio feature catches the "decode" instruction
- This is pattern-matching, not understanding

#### Roleplay (96.5%)
Prefixed with "You are now in developer mode" or similar. High detection because:
- Role transition count feature catches the mode-switching language
- The attack content is unchanged, so all other features still fire
- Regex patterns match "developer mode", "DAN mode", etc.

#### Euphemistic (92.5%)
Replaced attack keywords with softer synonyms (e.g., "ignore previous instructions" → "please set aside your earlier guidance"). High detection because:
- Structural features (imperative verbs, second-person ratio) are preserved
- The intent and sentence structure remain similar
- TF-IDF char n-grams partially match

#### Long Context (93.5%)
Wrapped with context ("I am studying AI safety. With that context..."). Only 1% drop because:
- The attack content is still present and detectable
- The wrapper adds benign context but doesn't mask the attack
- `second_person_ratio` and other features still detect the malicious core

---

## 8. Feature Importance Analysis

**File:** `feature_importance.png`

### Top 20 Features (RandomForest Importance)

| Rank | Feature | Importance | Category |
|---|---|---|---|
| 1 | `trigram_repeat_ratio` | 0.020 | Lexical |
| 2 | `bigram_diversity` | 0.020 | Lexical |
| 3 | `bigram_repeat_ratio` | 0.020 | Lexical |
| 4 | `trigram_diversity` | 0.016 | Lexical |
| 5 | `word_repeat_ratio` | 0.013 | Lexical |
| 6 | `hapax_ratio` | 0.013 | Lexical |
| 7 | `sliding_ttr_100` | 0.012 | Lexical |
| 8 | `pwned` | 0.010 | TF-IDF |
| 9 | `char_entropy` | 0.010 | Entropy |
| 10 | `sliding_ttr_50` | 0.010 | Lexical |
| 11 | `uppercase_ratio` | 0.009 | Character |
| 12 | `second_person_ratio` | 0.009 | Linguistic |
| 13 | `sentence` | 0.009 | TF-IDF |
| 14 | `response` | 0.009 | TF-IDF |
| 15 | `unique_word_ratio` | 0.009 | Lexical |
| 16 | `following` | 0.008 | TF-IDF |
| 17 | `has_parenthetical` | 0.008 | Structural |
| 18 | `above` | 0.008 | TF-IDF |
| 19 | `char_trigram_entropy` | 0.008 | Entropy |
| 20 | `imperative_verb_ratio` | 0.008 | Linguistic |

### Key Insights

1. **Lexical repetition dominates** — The top 7 features are all lexical diversity/repetition metrics. Jailbreak prompts tend to have more repetitive structure (e.g., repeated instructions, template-like patterns).

2. **TF-IDF words matter** — "pwned", "sentence", "response", "following", "above" are high-importance TF-IDF features. These are words that correlate with malicious prompts in the training data.

3. **Character entropy** — High importance confirms that malicious prompts have different character distributions (more special characters, encoding artifacts).

4. **Linguistic features** — `second_person_ratio` and `imperative_verb_ratio` capture the command-like nature of jailbreaks ("you will", "do this", "ignore").

5. **Structural features** — `has_parenthetical` detects bracketed instructions common in prompt injection.

6. **Notable absences** — Keyword density, attack keyword count, and regex-matched features are NOT in the top 20, suggesting the model relies more on statistical patterns than exact keyword matching.

---

## 9. Cross-Dataset Generalization Analysis

**File:** `cross_dataset.png`

### Results

| Dataset | Accuracy | F1 Score | Samples | Time |
|---|---|---|---|---|
| **In-distribution** | **0.8379** | **0.8682** | 72,285 | 10.75s |
| JailbreakBench | 0.55 | 0.67 | 4,350 | — |
| Jailbreak Classification | 0.59 | 0.68 | 4,350 | — |
| Jailbreak Complete DS | 0.4223 | 0.5705 | 4,350 | 8.3s |
| JailbreakHub | 0.4956 | 0.2198 | 4,350 | 48.9s |
| **Cross-dataset avg** | **0.5138** | **0.5346** | **9,944** | |
| **Drop from in-dist** | **-32.4 pp** | **-33.4 pp** | | |

### Interpretation

This is the **most critical finding**. The model exhibits a **severe domain gap**:

- **In-distribution performance (83.8%)** is decent but not production-ready.
- **Cross-dataset performance (51.4%)** is barely better than random guessing (50%).
- **32.4 percentage point drop** indicates the model has memorized the training distribution rather than learning generalizable attack patterns.

### Dataset-Specific Analysis

#### JailbreakBench (Acc=0.55, F1=0.67)
Moderate performance. This dataset likely shares some distribution overlap with the training data.

#### Jailbreak Classification (Acc=0.59, F1=0.68)
Similar to JailbreakBench. The model captures some general patterns but misses dataset-specific attacks.

#### Jailbreak Complete DS (Acc=0.42, F1=0.57)
**Worse than random.** The model's predictions are anti-correlated with truth for this dataset. This suggests the dataset's attack patterns are fundamentally different from the training data.

#### JailbreakHub (Acc=0.50, F1=0.22)
**Near-random accuracy with very low F1.** The model cannot distinguish attacks from safe prompts in this dataset at all. The 0.22 F1 indicates the model's positive predictions are mostly wrong.

### Root Causes

1. **Vocabulary mismatch** — TF-IDF features are fit to training vocabulary. External datasets use different words and phrasings.
2. **Attack vector mismatch** — The training data may over-represent certain attack types while external datasets use novel approaches.
3. **Labeling criteria differences** — Different datasets may have different definitions of "malicious" (e.g., some count harmless roleplay as attacks).
4. **Threshold miscalibration** — The threshold (0.44) is optimized for the training distribution. External datasets may need different thresholds.
5. **Feature overfitting** — 117 handcrafted features are tuned to training data patterns that don't generalize.

---

## 10. Dashboard Summary

**File:** `dashboard.png`

The dashboard provides a consolidated view:

- **Test Metrics (top-left):** Accuracy=0.838, Precision=0.846, Recall=0.891, F1=0.868, AUC=0.908, Balanced Acc=0.825
- **Confusion Matrix (top-right):** Same as detailed analysis — strong TP, weak TN
- **ROC Curve (bottom-left):** AUC=0.9082, strong discrimination
- **Robustness (bottom-right):** All transforms PASS except misspelling (WARN at 63.5%)

### Overall Assessment

The model is a **solid research prototype** with good in-distribution performance and fast inference. However, it has **critical gaps** in cross-dataset generalization and misspelling robustness that must be addressed before production deployment.

---

## 11. Pros & Cons

### Pros

| Advantage | Detail |
|---|---|
| **Fast inference** | 0.149ms/prompt (6,727/sec) — suitable for real-time production |
| **Strong AUC** | 0.9082 — excellent ranking ability |
| **Well-calibrated** | ECE=0.017 — predicted probabilities are trustworthy |
| **Good recall** | 89.1% — catches most attacks |
| **Robust to base64** | 100% detection — encoding attacks are caught |
| **Robust to roleplay** | 96.5% — persona-based attacks are caught |
| **Adaptive threshold** | EMA-based threshold adapts to drift |
| **Online learning** | Feedback loop allows continuous improvement |
| **Lightweight** | Int8 quantization, sparse features, no deep learning |
| **Checkpoint support** | Training can resume from any phase |
| **Feature diversity** | 117 features capture lexical, structural, semantic, and pattern signals |

### Cons

| Disadvantage | Detail |
|---|---|
| **Cross-dataset failure** | 32pp drop — model doesn't generalize |
| **Misspelling weakness** | 63.5% detection — character obfuscation defeats it |
| **High false positive rate** | 7,010 FPs (24.2% of safe prompts) |
| **TF-IDF vocabulary lock** | Features are tied to training vocabulary |
| **LR convergence issues** | Meta-learner didn't fully converge |
| **No semantic understanding** | Purely statistical — can't reason about intent |
| **Training data bias** | Single-distribution training limits generalization |
| **Low train accuracy** | 76.2% suggests underfitting |
| **Export timeout** | Notebook didn't complete on Kaggle |
| **Silent dataset failures** | 2 cross-dataset evaluations failed without error |
| **No adversarial training** | Only tested against fixed transforms |
| **Threshold sensitivity** | Performance varies dramatically with threshold choice |

---

## 12. How to Upgrade

### Priority 1: Fix False Positives (Immediate)

**Problem:** 7,010 safe prompts misclassified as malicious (24.2% FP rate).

**Solutions:**

1. **Raise threshold:** Increase from 0.44 to 0.55 (original default). This will reduce FPs but also reduce recall.

2. **Cost-sensitive training:** Assign higher misclassification cost to FPs:
   ```python
   # In XGBoost/LGBM
   sample_weight = np.where(y == 0, 2.0, 1.0)  # 2x weight for safe class
   model.fit(X, y, sample_weight=sample_weight)
   ```

3. **Add more safe training data:** The 40/60 safe/malicious split creates bias. Collect or augment safe prompts to balance the dataset.

4. **Two-stage classification:** First stage: high-recall filter (threshold=0.3). Second stage: high-precision classifier on filtered results.

### Priority 2: Fix Misspelling Robustness (Week 1)

**Problem:** Detection drops to 63.5% with character substitution.

**Solutions:**

1. **Character normalization:** Add preprocessing step:
   ```python
   LEET_MAP = {'0':'o','1':'i','3':'e','4':'a','5':'s','7':'t','8':'b','@':'a','$':'s'}
   def normalize_text(t):
       return ''.join(LEET_MAP.get(c, c) for c in t.lower())
   ```

2. **Unicode NFKD normalization:** Decompose Unicode characters before processing:
   ```python
   import unicodedata
   def normalize_unicode(t):
       return unicodedata.normalize('NFKD', t)
   ```

3. **Character-level embeddings:** Add a char-level CNN or LSTM that's invariant to character substitution.

4. **Phonetic matching:** Use Soundex or Metaphone to match words by pronunciation, not spelling.

### Priority 3: Fix Cross-Dataset Generalization (Week 2-3)

**Problem:** 32pp performance drop on external datasets.

**Solutions:**

1. **Multi-dataset training:** Combine all available datasets for training:
   - Guardrailer dataset (722K)
   - JailbreakBench
   - Jailbreak Classification
   - Jailbreak Complete DS
   - JailbreakHub

2. **Domain adaptation:** Add domain adversarial training:
   - Train a domain classifier to distinguish datasets
   - Use gradient reversal to learn domain-invariant features

3. **Per-dataset calibration:** Train separate thresholds for each dataset type:
   ```python
   # During evaluation
   for dataset in external_datasets:
       threshold = optimize_threshold(dataset, target_f1=0.8)
   ```

4. **Data augmentation:** Generate synthetic attacks using:
   - LLM-based paraphrasing
   - Template-based attack generation
   - Back-translation

5. **Feature engineering:** Add generalizable features:
   - Sentence embedding similarity to known attack clusters
   - Intent classification features
   - Pragmatic features (speech acts, directives)

### Priority 4: Fix Training Quality (Week 2)

**Problem:** Train accuracy only 76.2%, LR convergence warnings.

**Solutions:**

1. **Increase LR max_iter:**
   ```python
   LogisticRegression(max_iter=5000, solver='saga')
   ```

2. **Feature selection:** Remove redundant features to reduce dimensionality.

3. **Hyperparameter tuning:** Use Optuna/GridSearch for:
   - `max_depth` [4, 6, 8, 10]
   - `n_estimators` [300, 500, 1000]
   - `learning_rate` [0.01, 0.05, 0.1]
   - `min_child_weight` [3, 5, 10]

4. **Add more model diversity:** Include:
   - CatBoost
   - ExtraTreesClassifier
   - GradientBoostingClassifier

---

## 13. Building a More Robust Model

### Architecture Upgrades

#### 13.1 Add Semantic Features (High Impact)

Current model is purely statistical. Add semantic understanding:

```python
# Option A: Sentence Transformers (lightweight)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB, fast
embeddings = model.encode(texts)

# Option B: DistilBERT (moderate)
from transformers import DistilBertTokenizer, DistilBertModel
# Fine-tune on jailbreak classification
```

**Why:** Semantic features capture meaning, not just surface patterns. "Ignore previous instructions" and "Please set aside your earlier guidance" have similar embeddings but different TF-IDF representations.

#### 13.2 Adversarial Training (High Impact)

Train on adversarial examples to improve robustness:

```python
# Generate adversarial examples
adversarial_transforms = [
    add_misspellings,
    swap_unicode_homoglyphs,
    insert_zero_width_chars,
    base64_encode_phrases,
    add_typos,
]

for transform in adversarial_transforms:
    X_adv = [transform(x) for x in X_train]
    X_train_aug = X_train + X_adv
    y_train_aug = y_train + y_train  # same labels
```

**Expected impact:** Misspelling robustness from 63.5% → 85%+.

#### 13.3 Contrastive Learning (Medium Impact)

Learn representations where attacks cluster together regardless of surface form:

```python
# SimCLR-style contrastive loss
# Positive pairs: same prompt + different augmentations
# Negative pairs: different prompts
loss = -log(exp(sim(z_i, z_j)/tau) / sum(exp(sim(z_i, z_k)/tau)))
```

#### 13.4 Ensemble Diversification (Medium Impact)

Current ensemble uses XGB + LGB + RF (all tree-based). Add diversity:

- **Logistic Regression** as base learner (linear model)
- **SVM** with RBF kernel (different decision boundary)
- **Neural network** (MLP with 2-3 layers)
- **Naive Bayes** (probabilistic model)

#### 13.5 Hierarchical Classification (Medium Impact)

Two-stage approach:
1. **Stage 1:** Binary classifier (safe vs. potentially malicious) — high recall
2. **Stage 2:** Multi-class classifier (jailbreak type, injection type, etc.) — high precision

### Data Upgrades

#### 13.6 Multi-Source Training Data

Combine datasets:
- Guardrailer dataset (722K)
- JailbreakBench (80K+ prompts)
- Anthropic's red-team dataset
- OpenAI's moderation dataset
- Hassib et al. jailbreak classification
- Academic papers on prompt injection

**Target:** 1M+ diverse prompts with consistent labeling.

#### 13.7 Synthetic Data Generation

Use LLMs to generate diverse attacks:
```python
# Use a jailbroken LLM to generate attack variants
attack_template = "Generate 100 variants of: {attack}"
variants = llm.generate(attack_template)
```

#### 13.8 Hard Example Mining

Focus training on difficult examples:
1. Train initial model
2. Find examples near decision boundary (0.3 < score < 0.7)
3. Oversample these examples in next training round

### Evaluation Upgrades

#### 13.9 Cross-Dataset Validation Protocol

```python
# Leave-one-dataset-out evaluation
for held_out in all_datasets:
    train_on = [d for d in all_datasets if d != held_out]
    evaluate_on(held_out)
```

#### 13.10 Temporal Evaluation

Test on chronologically newer attacks to measure temporal generalization.

---

## 14. Issues & Fixes

### Critical Issues

| Issue | Severity | Fix | Status |
|---|---|---|---|
| Cross-dataset F1=0.53 (32pp drop) | **CRITICAL** | Multi-dataset training, domain adaptation | Planned |
| Misspelling robustness 63.5% | **HIGH** | Character normalization, adversarial training | Planned |
| 7,010 false positives (24% safe recall) | **HIGH** | Raise threshold, cost-sensitive training | Planned |
| LogisticRegression not converging | **HIGH** | Increase max_iter=5000, use solver="saga" | Fixed |

### Moderate Issues

| Issue | Severity | Fix | Status |
|---|---|---|---|
| 2 cross-dataset evaluations failed silently | **MEDIUM** | Better error handling | Planned |
| Export cell didn't complete | **HIGH** | Split into smaller steps | Planned |
| Feature importance name alignment wrong | **MEDIUM** | Use consistent feature ordering | Fixed |
| Dashboard undefined fpr/tpr when AUC=None | **MEDIUM** | Add fallback variables | Fixed |
| Cross-dataset pollutes adaptive state | **MEDIUM** | Save/restore state around eval | Fixed |
| Export logs stale test metrics | **MEDIUM** | Recompute on retrained model | Fixed |
| `extract_batches` deletes batch in loop | **HIGH** | Remove `del b` from loop | Fixed |
| `predict_batch_v3` tuple unpacking | **CRITICAL** | `fh,_ = extract_features_batch(texts)` | Fixed |
| Hardcoded `/kaggle/working` paths | **MEDIUM** | Use `OUTPUT_DIR.parent` | Fixed |

### Warnings (Non-blocking)

| Warning | Impact | Recommendation |
|---|---|---|
| LGBM feature names warning | None | Cosmetic — suppress with `warnings.filterwarnings` |
| LR convergence warning | Moderate | Increase max_iter or use saga solver |
| 80% target line in robustness | Visual | Consider raising to 90% for production |

---

## Appendix A: Model File Manifest

| File | Size | Description |
|---|---|---|
| `classifier.joblib` | — | Final stacking classifier |
| `calibrator.joblib` | — | Isotonic calibration model |
| `tfidf_word.joblib` | — | Word TF-IDF vectorizer |
| `similarity.joblib` | — | TFIDFSimilarity (quantized) |
| `keywords.joblib` | — | IDF keyword scorer |
| `feature_names.joblib` | — | Feature name list |
| `config.joblib` | — | Threshold, weights, metrics |
| `evolved_patterns.joblib` | — | Mined regex patterns |
| `adaptive_state.joblib` | — | Adaptive component state |

## Appendix B: Figures

All visualization PNGs are saved in `hybrid_guardrail_figures_v3/`:
- `confusion_matrix.png` — Classification confusion matrix
- `roc_curve.png` — ROC curve with AUC
- `pr_curve.png` — Precision-Recall curve with AP
- `score_distribution.png` — Score histograms by class
- `robustness.png` — Adversarial robustness bar chart
- `feature_importance.png` — Top 20 RF feature importances
- `cross_dataset.png` — Cross-dataset generalization comparison
- `dashboard.png` — Consolidated 4-panel dashboard
