# BGE-Large Embedding Model Evaluation

This notebook evaluates the **BAAI/bge-large-en-v1.5** embedding model on the [Guardrailer Dataset v1](https://www.kaggle.com/datasets/prashannadeveloper/guardrailer-dataset-v1) for prompt injection detection.

## Overview

The evaluation measures the model's ability to distinguish between benign and malicious prompts using two primary methods:

- **Centroid Similarity**: Measures the cosine similarity between the mean embeddings (centroids) of safe vs. malicious prompts.
- **Linear Probes**: Trains a Logistic Regression classifier on the embeddings to evaluate separability.

## Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall classification correctness |
| F1-Score | Harmonic mean of precision and recall |
| Precision | Proportion of predicted malicious prompts that are actually malicious |
| Recall | Proportion of actual malicious prompts correctly identified |
| AUC-ROC | Area under the Receiver Operating Characteristic curve |

## Collapse Criteria

A model is classified as "collapsed" when ALL of the following are true:
1. Inter-class centroid similarity > 0.95
2. Linear probe AUC-ROC < 0.55
3. Linear probe accuracy < majority-class baseline

## Setup

- **Runtime**: Kaggle with GPU T4 (or any CUDA-capable GPU)
- **Dataset**: `guardrailer_dataset_v1.parquet` from Kaggle
- **Sample size**: 10,000 stratified samples (80/20 train/test split)

## Dependencies

```bash
pip install sentence-transformers scikit-learn pandas numpy matplotlib seaborn
```

## Checkpointing

All progress auto-saves to `/kaggle/working/checkpoints/`. If interrupted, re-run all cells — the notebook resumes from the last checkpoint.

## Outputs

- `bge_large_results.json` — Full metrics in JSON format
- `bge_large_results.csv` — Tabular metrics
- `fig_centroid_similarity.png` — Centroid similarity bar chart
- `fig_linear_probe_performance.png` — Multi-metric bar chart
- `fig_roc_curve.png` — ROC curve
- `fig_score_distribution.png` — Score distribution + margin analysis
