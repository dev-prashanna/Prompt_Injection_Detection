# Prompt Injection Detection

A research project investigating prompt injection detection using embedding-based scoring and lightweight statistical features. Published as a LaTeX paper with full experimental code.

## Overview

This project systematically investigates why nearest-centroid scoring fails on embedding representations for prompt injection detection, and develops a lightweight hybrid guardrail achieving 83.0% in-distribution accuracy with 0.149ms inference latency.

### Key Findings

- **Embedding space contains discriminative information** — linear probes achieve AUC 0.844–0.884
- **Nearest-centroid scoring fails** — 4-class centroid overlap ($C = 0.986$) yields AUC 0.547
- **TF-IDF alone outperforms full hybrid** — ablation study shows handcrafted features add noise
- **Severe generalization gap** — 83.0% in-distribution → 51.4% on external benchmarks

## Repository Structure

```
Prompt_Injection_Detection/
├── notebooks/
│   ├── centroid_evaluation/       # Centroid-based scoring experiments
│   │   ├── near_centroid_cosine.ipynb
│   │   ├── near_centroid_evaluation.ipynb
│   │   ├── near_centroid_standardscaler.ipynb
│   │   └── notebook5b0eb3a656.ipynb
│   ├── hybrid_guardrail_v3/       # Hybrid guardrail training
│   │   ├── train_v3_colab.ipynb
│   │   └── train_v3_kaggle_p100.ipynb
│   └── ablation_study/            # Feature ablation experiments
│       └── abalation_study.ipynb
├── src/                           # Python source files
│   ├── build_notebook.py
│   ├── build_notebook_p100.py
│   ├── build_ablation_notebook.py
│   └── evaluate_phase6_full.py
├── data/
│   ├── centroid_results/          # Centroid scoring results (CSV/JSON)
│   ├── phase6_results/            # Phase 6 multi-signal report
│   └── phase6_checkpoints/        # Phase 6 pipeline checkpoints
├── figures/                       # Generated figures
│   ├── centroid_evaluation/       # Centroid similarity, ROC curves, etc.
│   └── hybrid_guardrail/          # Confusion matrix, feature importance, etc.
├── paper/                         # LaTeX manuscript
│   ├── prompt_injection_detection.tex
│   ├── references.bib
│   └── prompt_injection_detection.pdf
├── docs/                          # Documentation and analysis
│   ├── EXPERIMENT_REPORT.md
│   ├── MANUSCRIPT.md
│   └── ...
├── multiple_embedding_model_testing/  # Multi-model comparison
│   ├── near_centroid_cosine.ipynb
│   ├── near_centroid_evaluation.ipynb
│   ├── README.md
│   └── centroid_failure_analysis.md
├── requirements.txt
└── README.md
```

## Experimental Pipeline

### Phase 6: Multi-Signal Evaluation
1. **Data Loading** — 722,842 samples (80/10/10 split)
2. **Centroid Computation** — 4 category centroids in embedding space
3. **Centroid Scoring** — Cosine similarity to centroids
4. **Sparse Scoring** — TF-IDF and text-based features
5. **Text Signal Extraction** — Regex patterns, entropy, statistical features
6. **Composite Scoring** — Weighted combination of signals
7. **Classification** — Threshold-based decision

### Hybrid Guardrail V3
- **Architecture**: Stacking ensemble (Random Forest + Linear SVM → Logistic Regression)
- **Features**: 117 handcrafted security features + TF-IDF (5000 features, int8 quantized)
- **Performance**: 83.0% accuracy, 0.827 F1, 0.908 AUC (in-distribution)

### Ablation Study
| Configuration | Accuracy | F1 Score |
|---|---|---|
| TF-IDF Only | 85.0% | 0.846 |
| Handcrafted Only | 71.9% | 0.684 |
| Full Hybrid (117 + TF-IDF) | 83.0% | 0.827 |

## Installation

```bash
git clone https://github.com/dev-prashanna/Prompt_Injection_Detection.git
cd Prompt_Injection_Detection
pip install -r requirements.txt
```

## Usage

1. **Centroid Evaluation**: Run notebooks in `notebooks/centroid_evaluation/`
2. **Hybrid Guardrail Training**: Run `notebooks/hybrid_guardrail_v3/train_v3_colab.ipynb`
3. **Ablation Study**: Run `notebooks/ablation_study/abalation_study.ipynb`

## Results

### Cross-Dataset Generalization
| Dataset | Accuracy | Notes |
|---|---|---|
| In-Distribution Test | 83.0% | Evaluated split |
| Average External | 51.4% | 4 external benchmarks |
| Jailbreak Complete DS | 42.2% | Below 0.50 reference level |

### Robustness to Transformations
- 7 of 8 fixed transformations maintain ≥80% detection
- Critical weakness: misspelling (63.5%)

## Citation

If you use this code or paper in your research, please cite:

```bibtex
@article{tiwari2025prompt,
  title={Lightweight Prompt Injection Detection: From Centroid-Based Embedding Classification to a Hybrid Guardrail},
  author={Tiwari, Prashanna},
  year={2025}
}
```

## License

This project is for research purposes. Please contact the author for usage permissions.
