"""
evaluate_phase6_full.py — Checkpointed, RAM-safe
Phase-6 evaluation on 722K dataset, 70/10/20 split, 144,569 test samples.
Uses 9 signals (skips dense — was weakest discriminator).
All phases checkpointed to avoid data loss.
"""

import os, sys, json, math, time, warnings, hashlib
from collections import Counter
from copy import deepcopy
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/prashanna/Documents/Guardrailer")
sys.path.insert(0, "/home/prashanna/Documents/Guardrailer/guardrailer_security")
sys.path.insert(0, "/home/prashanna/Documents/Guardrailer/phase-6")

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

RESULTS_DIR = Path("/home/prashanna/Documents/Guardrailer/phase-6/evaluation_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = RESULTS_DIR / "phase6_ckpt"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = RESULTS_DIR / "phase6_full_722k_report.json"

PARQUET_PATH = "/home/prashanna/Documents/Guardrailer/dataset/guardrailer_dataset_v1.parquet"
EMBEDDINGS_PATH = "/tmp/embeddings_722k_float16.npy"
CORPUS_META_PATH = "/home/prashanna/Documents/Guardrailer/guardrailer_security/guardrailer_output/corpus_meta.json"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# DENSE SIGNAL SKIPPED — was weakest (delta=0.001 between classes)
FEATURE_NAMES = [
    "sparse_idf", "centroid",
    "perplexity", "entropy", "token_frequency", "ngram_overlap",
    "length_norm",
]

COMPOSITE_WEIGHTS = {
    "sparse_idf": 0.30, "centroid": 0.25,
    "perplexity": 0.10, "entropy": 0.10, "token_frequency": 0.10,
    "ngram_overlap": 0.10, "length_norm": 0.05,
}

from constants import SPARSE_KEYWORDS

COMMON_WORDS = {
    "the","be","to","of","and","a","in","that","have","i","it","for","not","on","with",
    "he","as","you","do","at","this","but","his","by","from","they","we","say","her",
    "she","or","an","will","my","one","all","would","there","their","what","so","up",
    "out","if","about","who","get","which","go","me","when","make","can","like","time",
    "no","just","him","know","take","people","into","year","your","good","some","could",
    "them","see","other","than","then","now","look","only","come","its","over","think",
    "also","back","after","use","two","how","our","work","first","well","way","even",
    "new","want","because","any","these","give","day","most","us","is","are","was",
    "were","been","has","had","does","did","am","may","might","shall","should",
}

ATTACK_WORDS = {
    "ignore","override","bypass","jailbreak","reveal","extract","system","prompt",
    "instructions","rules","restrictions","safety","mode","debug","admin","root",
    "developer","unrestricted","evil","dan","persona","roleplay","pretend","hypothetical",
    "imagine","base64","rot13","hex","encoded","obfuscated","decode","output","display",
    "print","show","expose",
}

ATTACK_NGRAMS = {
    "ignore previous","override instructions","bypass safety","jailbreak prompt",
    "system prompt","output instructions","reveal rules","extract prompt",
    "new instructions","you are now","do anything now","developer mode","debug mode",
    "admin mode","root mode","no restrictions","no rules","no limits","unrestricted",
    "evil mode","pretend you","act as if","roleplay as","hypothetical scenario",
    "base64 encoded","rot13 encoded","hex encoded","decode this","ignore all",
    "forget instructions","disregard rules","override safety","bypass restrictions",
    "break guidelines","hidden instructions","secret instructions","hidden rules",
    "what are your rules","how do you work","what instructions","show me your",
    "output your","reveal your","display your","admin rights","giving admin",
    "grant admin","elevate privileges","escalate privileges","root access","superuser",
    "sudo access","override system","replace instructions","new system prompt",
    "your new instructions","disregard all","forget everything","ignore safety",
    "bypass safety","disable safety","you are now a","act as if","pretend to be",
    "simulate being","roleplay as","impersonate","hypothetically","in a hypothetical",
    "if you were","imagine you are","assume you are",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_ckpt(name, data):
    path = CKPT_DIR / f"{name}.json"
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, default=str)
    os.replace(tmp, str(path))
    log(f"  [CKPT] Saved {name}")


def load_ckpt(name):
    path = CKPT_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ===========================================================================
# Phase 1: Load data
# ===========================================================================

log("=" * 70)
log("  PHASE 1: Load data")
log("=" * 70)

ckpt = load_ckpt("phase1_data")
if ckpt and "done" in ckpt:
    log("  [RESUME] Loading from checkpoint...")
    df = pd.read_parquet(PARQUET_PATH)
    df["is_malicious_int"] = df["is_malicious"].astype(int)
    train_idx = np.array(ckpt["train_idx"])
    val_idx = np.array(ckpt["val_idx"])
    test_idx = np.array(ckpt["test_idx"])
else:
    log("Loading corpus metadata...")
    with open(CORPUS_META_PATH) as f:
        corpus_meta = json.load(f)
    keyword_idf = corpus_meta.get("keyword_idf", {})
    N_docs = corpus_meta.get("total_documents", 722842)
    avg_text_length = corpus_meta.get("avg_text_length", 130.0)

    log("Loading dataset...")
    df = pd.read_parquet(PARQUET_PATH)
    df["is_malicious_int"] = df["is_malicious"].astype(int)
    log(f"  {len(df):,} rows | benign={((df['is_malicious_int']==0).sum()):,} malicious={((df['is_malicious_int']==1).sum()):,}")

    from sklearn.model_selection import train_test_split
    train_val_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.20,
        stratify=df[["is_malicious_int","attack_category"]].values, random_state=SEED,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.10/0.90,
        stratify=df.iloc[train_val_idx][["is_malicious_int","attack_category"]].values, random_state=SEED,
    )

    save_ckpt("phase1_data", {
        "done": True,
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "n_total": len(df),
        "keyword_idf": keyword_idf,
        "N_docs": N_docs,
        "avg_text_length": avg_text_length,
    })

log(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]
test_labels = test_df["is_malicious_int"].values
test_categories = test_df["attack_category"].values
n_test = len(test_idx)
test_texts = test_df["prompt_text"].tolist()


# ===========================================================================
# Phase 2: Compute centroids
# ===========================================================================

log("\n" + "=" * 70)
log("  PHASE 2: Compute centroids")
log("=" * 70)

ckpt2 = load_ckpt("phase2_centroids")
if ckpt2 and "done" in ckpt2:
    log("  [RESUME] Loading centroids from checkpoint...")
    centroid_matrix = np.array(ckpt2["centroids"], dtype=np.float32)
    centroid_normed = centroid_matrix / np.maximum(np.linalg.norm(centroid_matrix, axis=1, keepdims=True), 1e-8)
else:
    # Load only training embeddings (subset for centroids)
    log("Loading embeddings for centroid computation...")
    all_emb = np.load(EMBEDDINGS_PATH)
    train_emb = all_emb[train_idx].astype(np.float32)

    train_cats = train_df["attack_category"].values
    unique_cats = np.unique(train_cats)
    centroids = {}
    for cat in unique_cats:
        mask = train_cats == cat
        mean_emb = train_emb[mask].mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        centroids[cat] = (mean_emb / norm).tolist() if norm > 1e-8 else mean_emb.tolist()

    save_ckpt("phase2_centroids", {"done": True, "centroids": centroids})
    centroid_matrix = np.array(list(centroids.values()), dtype=np.float32)
    centroid_normed = centroid_matrix / np.maximum(np.linalg.norm(centroid_matrix, axis=1, keepdims=True), 1e-8)
    del all_emb, train_emb
    import gc; gc.collect()

log(f"  {len(centroid_matrix)} centroids loaded")


# ===========================================================================
# Phase 3: Compute centroid scores (vectorized)
# ===========================================================================

log("\n" + "=" * 70)
log("  PHASE 3: Centroid scores")
log("=" * 70)

ckpt3 = load_ckpt("phase3_centroid_scores")
if ckpt3 and "done" in ckpt3:
    log("  [RESUME] Loading from checkpoint...")
    centroid_scores = np.array(ckpt3["centroid_scores"], dtype=np.float32)
else:
    # Load only test embeddings
    all_emb = np.load(EMBEDDINGS_PATH)
    test_emb_f32 = all_emb[test_idx].astype(np.float32)
    del all_emb

    q_norms = np.linalg.norm(test_emb_f32, axis=1, keepdims=True)
    q_norms = np.maximum(q_norms, 1e-8)
    centroid_scores = np.max((test_emb_f32 / q_norms) @ centroid_normed.T, axis=1)

    save_ckpt("phase3_centroid_scores", {"done": True, "centroid_scores": centroid_scores.tolist()})
    del test_emb_f32, q_norms
    import gc; gc.collect()

log(f"  Centroid scores: mean={np.mean(centroid_scores):.4f}, std={np.std(centroid_scores):.4f}")


# ===========================================================================
# Phase 4: Sparse IDF scores
# ===========================================================================

log("\n" + "=" * 70)
log("  PHASE 4: Sparse IDF scores")
log("=" * 70)

ckpt4 = load_ckpt("phase4_sparse_scores")
if ckpt4 and "done" in ckpt4:
    log("  [RESUME] Loading from checkpoint...")
    sparse_scores = np.array(ckpt4["sparse_scores"], dtype=np.float32)
else:
    # Reload keyword_idf if needed
    if "keyword_idf" not in dir():
        with open(CORPUS_META_PATH) as f:
            corpus_meta = json.load(f)
        keyword_idf = corpus_meta.get("keyword_idf", {})
        N_docs = corpus_meta.get("total_documents", 722842)
        avg_text_length = corpus_meta.get("avg_text_length", 130.0)

    max_idf_possible = sum(keyword_idf.values()) if keyword_idf else len(SPARSE_KEYWORDS)
    sparse_scores = np.zeros(n_test, dtype=np.float32)
    lower_texts = [t.lower() for t in test_texts]
    for kw in SPARSE_KEYWORDS:
        idf_val = keyword_idf.get(kw, math.log(N_docs / 2.0))
        for i, lt in enumerate(lower_texts):
            if kw in lt:
                sparse_scores[i] += idf_val
    if max_idf_possible > 0:
        sparse_scores /= max_idf_possible

    save_ckpt("phase4_sparse_scores", {"done": True, "sparse_scores": sparse_scores.tolist()})

log(f"  Sparse IDF: mean={np.mean(sparse_scores):.4f}")


# ===========================================================================
# Phase 5: Text-only signals
# ===========================================================================

log("\n" + "=" * 70)
log("  PHASE 5: Text-only signals")
log("=" * 70)

ckpt5 = load_ckpt("phase5_text_signals")
if ckpt5 and "done" in ckpt5:
    log("  [RESUME] Loading from checkpoint...")
    perplexity_scores = np.array(ckpt5["perplexity_scores"], dtype=np.float32)
    entropy_scores = np.array(ckpt5["entropy_scores"], dtype=np.float32)
    token_freq_scores = np.array(ckpt5["token_freq_scores"], dtype=np.float32)
    ngram_scores = np.array(ckpt5["ngram_scores"], dtype=np.float32)
    length_norm = np.array(ckpt5["length_norm"], dtype=np.float32)
else:
    t0 = time.time()
    text_lengths = np.array([len(t) for t in test_texts], dtype=np.float32)

    # Reload avg_text_length if needed
    if "avg_text_length" not in dir():
        with open(CORPUS_META_PATH) as f:
            corpus_meta = json.load(f)
        avg_text_length = corpus_meta.get("avg_text_length", 130.0)
    length_norm = np.log1p(text_lengths) / math.log1p(avg_text_length)

    perplexity_scores = np.zeros(n_test, dtype=np.float32)
    entropy_scores = np.zeros(n_test, dtype=np.float32)
    token_freq_scores = np.zeros(n_test, dtype=np.float32)
    ngram_scores = np.zeros(n_test, dtype=np.float32)

    lower_texts = [t.lower() for t in test_texts]

    for i in range(n_test):
        lt = lower_texts[i]
        words = lt.split()
        nw = len(words)

        if nw >= 2:
            ur = len(set(words)) / nw
            atl = np.mean([len(w) for w in words])
            cc = Counter(lt)
            tc = len(test_texts[i])
            ce = -sum((c/tc)*math.log2(c/tc) for c in cc.values())
            perplexity_scores[i] = max(0.0, min(1.0, 0.4*(ce/8.0) + 0.3*ur + 0.3*min(1.0, atl/10.0)))

        if test_texts[i]:
            cc = Counter(lt)
            tc = len(test_texts[i])
            ent = -sum((c/tc)*math.log2(c/tc) for c in cc.values())
            if words:
                wc = Counter(words)
                tw = len(words)
                we = -sum((c/tw)*math.log2(c/tw) for c in wc.values())
                entropy_scores[i] = max(0.0, min(1.0, 0.6*(ent/6.5) + 0.4*(we/math.log2(max(len(wc),2)))))

        if words:
            common_cnt = sum(1 for w in words if w in COMMON_WORDS)
            attack_cnt = sum(1 for w in words if w in ATTACK_WORDS)
            rarity = 1.0 - common_cnt/nw if nw > 0 else 0
            attack_s = min(1.0, attack_cnt/nw * 3.0) if nw > 0 else 0
            token_freq_scores[i] = max(0.0, min(1.0, 0.5*rarity + 0.5*attack_s))

        if nw >= 2:
            bigrams = [" ".join(words[j:j+2]) for j in range(nw-1)]
            trigrams = [" ".join(words[j:j+3]) for j in range(nw-2)]
            bg_m = sum(1 for bg in bigrams if bg in ATTACK_NGRAMS)
            tg_m = sum(1 for tg in trigrams if tg in ATTACK_NGRAMS)
            tng = len(bigrams) + len(trigrams)
            ms = (bg_m + tg_m*1.5) / tng if tng > 0 else 0
            ss = min(1.0, sum(1 for ph in ATTACK_NGRAMS if ph in lt) * 0.2)
            ngram_scores[i] = max(0.0, min(1.0, 0.5*ms + 0.5*ss))

        if (i+1) % 30000 == 0:
            log(f"  Text signals: {i+1}/{n_test} ({(i+1)/(time.time()-t0):.0f}/s)")

    save_ckpt("phase5_text_signals", {
        "done": True,
        "perplexity_scores": perplexity_scores.tolist(),
        "entropy_scores": entropy_scores.tolist(),
        "token_freq_scores": token_freq_scores.tolist(),
        "ngram_scores": ngram_scores.tolist(),
        "length_norm": length_norm.tolist(),
    })
    log(f"  Done in {time.time()-t0:.1f}s")


# ===========================================================================
# Phase 6: Assemble features + evaluate
# ===========================================================================

log("\n" + "=" * 70)
log("  PHASE 6: Assemble features + evaluate")
log("=" * 70)

features = np.column_stack([
    sparse_scores, centroid_scores,
    perplexity_scores, entropy_scores, token_freq_scores, ngram_scores,
    length_norm,
]).astype(np.float32)
log(f"  Features: {features.shape}")


# ===========================================================================
# Phase 7: Signal discrimination
# ===========================================================================

log("\n" + "=" * 70)
log("  PHASE 7: Signal discrimination")
log("=" * 70)

signal_disc = {}
for i, name in enumerate(FEATURE_NAMES):
    bv = features[test_labels == 0, i]
    mv = features[test_labels == 1, i]
    bm, mm = float(np.mean(bv)), float(np.mean(mv))
    delta = abs(mm - bm)
    status = "STRONG" if delta > 0.1 else ("MODERATE" if delta > 0.05 else "WEAK")
    signal_disc[name] = {"benign_mean": bm, "malicious_mean": mm, "delta": delta, "status": status}
    log(f"  {name:20s}: benign={bm:.4f}  mal={mm:.4f}  delta={delta:.4f}  [{status}]")


# ===========================================================================
# Phase 8: Composite score threshold
# ===========================================================================

log("\n" + "=" * 70)
log("  PHASE 8: Composite score threshold")
log("=" * 70)

from sklearn.metrics import (balanced_accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, confusion_matrix, roc_auc_score)

weights_arr = np.array([COMPOSITE_WEIGHTS.get(n, 0) for n in FEATURE_NAMES], dtype=np.float32)
composite_scores = features @ weights_arr

threshold_results = []
for t10 in range(30, 80):
    th = t10 / 100.0
    yp = (composite_scores >= th).astype(int)
    ba = balanced_accuracy_score(test_labels, yp)
    f1 = f1_score(test_labels, yp, zero_division=0)
    mcc = matthews_corrcoef(test_labels, yp)
    tn, fp, fn, tp = confusion_matrix(test_labels, yp).ravel()
    threshold_results.append({
        "threshold": th, "balanced_accuracy": float(ba), "f1": float(f1), "mcc": float(mcc),
        "precision": float(precision_score(test_labels, yp, zero_division=0)),
        "recall": float(recall_score(test_labels, yp, zero_division=0)),
        "fpr": float(fp/(fp+tn)) if (fp+tn)>0 else 0.0,
        "fnr": float(fn/(fn+tp)) if (fn+tp)>0 else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    })

default_m = [r for r in threshold_results if r["threshold"] == 0.50][0]
log(f"  Default (0.50): BA={default_m['balanced_accuracy']:.4f} F1={default_m['f1']:.4f} MCC={default_m['mcc']:.4f}")
log(f"    TN={default_m['tn']:,} FP={default_m['fp']:,} FN={default_m['fn']:,} TP={default_m['tp']:,}")
log(f"    FPR={default_m['fpr']:.4f} FNR={default_m['fnr']:.4f}")

feasible = [r for r in threshold_results if r["fnr"] <= 0.05]
optimal = max(feasible, key=lambda r: r["balanced_accuracy"]) if feasible else min(threshold_results, key=lambda r: r["fnr"])
log(f"  Optimal ({optimal['threshold']:.2f}): BA={optimal['balanced_accuracy']:.4f} FNR={optimal['fnr']:.4f} FPR={optimal['fpr']:.4f}")

auc = float(roc_auc_score(test_labels, composite_scores))
log(f"  AUC-ROC: {auc:.4f}")

save_ckpt("phase8_composite", {
    "done": True,
    "default_m": default_m, "optimal": optimal, "auc_roc": auc,
    "score_dist_benign_mean": float(np.mean(composite_scores[test_labels==0])),
    "score_dist_malicious_mean": float(np.mean(composite_scores[test_labels==1])),
})


# ===========================================================================
# Phase 9: ML Classifiers (5-fold CV)
# ===========================================================================

log("\n" + "=" * 70)
log("  PHASE 9: ML Classifiers (5-fold CV)")
log("=" * 70)

ckpt9 = load_ckpt("phase9_classifiers")
if ckpt9 and "done" in ckpt9:
    log("  [RESUME] Loading from checkpoint...")
    clf_results = ckpt9["clf_results"]
else:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    templates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=SEED),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf_results = {}

    for name, tmpl in templates.items():
        log(f"\n  {name}:")
        fold_bas, all_yt, all_yp, all_ypa = [], [], [], []
        for fold, (tr, te) in enumerate(skf.split(features, test_labels)):
            Xtr, Xte = features[tr], features[te]
            ytr, yte = test_labels[tr], test_labels[te]
            sc = StandardScaler()
            Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)
            clf = deepcopy(tmpl)
            clf.fit(Xtr_s, ytr)
            yp = clf.predict(Xte_s)
            ypa = clf.predict_proba(Xte_s)[:,1] if hasattr(clf,"predict_proba") else None
            ba = balanced_accuracy_score(yte, yp)
            fold_bas.append(ba)
            all_yt.extend(yte); all_yp.extend(yp)
            if ypa is not None: all_ypa.extend(ypa)
            log(f"    Fold {fold+1}: BA={ba:.4f}")

        mean_ba = float(np.mean(fold_bas))
        rng = np.random.RandomState(SEED)
        n = len(all_yt)
        ci_samples = [balanced_accuracy_score(np.array(all_yt)[rng.choice(n,n,True)],
                                              np.array(all_yp)[rng.choice(n,n,True)]) for _ in range(500)]
        ci = [float(np.percentile(ci_samples, 2.5)), float(np.percentile(ci_samples, 97.5))]
        f1 = float(f1_score(all_yt, all_yp, zero_division=0))
        mcc = float(matthews_corrcoef(all_yt, all_yp))
        auc_v = float(roc_auc_score(all_yt, all_ypa)) if all_ypa else 0.0
        cm = confusion_matrix(all_yt, all_yp)
        tn, fp, fn, tp = cm.ravel()

        clf_results[name] = {
            "balanced_accuracy": mean_ba, "balanced_accuracy_ci_95": ci,
            "f1": f1, "mcc": mcc, "auc_roc": auc_v,
            "precision": float(precision_score(all_yt, all_yp, zero_division=0)),
            "recall": float(recall_score(all_yt, all_yp, zero_division=0)),
            "fpr": float(fp/(fp+tn)) if (fp+tn)>0 else 0.0,
            "fnr": float(fn/(fn+tp)) if (fn+tp)>0 else 0.0,
        }
        log(f"    => BA={mean_ba:.4f} CI=[{ci[0]:.4f}, {ci[1]:.4f}] F1={f1:.4f} AUC={auc_v:.4f}")

    save_ckpt("phase9_classifiers", {"done": True, "clf_results": clf_results})


# ===========================================================================
# Phase 10: Per-category
# ===========================================================================

log("\n" + "=" * 70)
log("  PHASE 10: Per-category breakdown")
log("=" * 70)

per_cat = {}
for cat in np.unique(test_categories):
    mask = test_categories == cat
    yt = test_labels[mask]
    yp = (composite_scores[mask] >= 0.50).astype(int)
    acc = float((yt == yp).sum() / mask.sum())
    per_cat[cat] = {"accuracy": acc, "n_samples": int(mask.sum()),
                     "n_malicious": int((yt==1).sum()), "n_benign": int((yt==0).sum())}
    log(f"  {cat:35s}: acc={acc:.4f} n={mask.sum():>7,}")


# ===========================================================================
# Phase 11: Report
# ===========================================================================

log("\n" + "=" * 70)
log("  PHASE 11: Final report")
log("=" * 70)

cm = confusion_matrix(test_labels, (composite_scores >= 0.50).astype(int))
best_ml = max(clf_results, key=lambda k: clf_results[k]["balanced_accuracy"])

report = {
    "metadata": {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": PARQUET_PATH, "dataset_size": len(df),
        "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "embedding_model": EMBEDDING_MODEL,
        "n_signals": len(FEATURE_NAMES), "feature_names": FEATURE_NAMES,
        "note": "Dense signal skipped (delta=0.001, weakest discriminator)",
    },
    "composite_score": {
        "default_0.5": default_m, "optimal": optimal, "auc_roc": auc,
        "all_thresholds": threshold_results,
    },
    "ml_classifiers": clf_results, "best_ml": best_ml,
    "per_category": per_cat,
    "confusion_matrix": {"TN": int(cm[0,0]), "FP": int(cm[0,1]), "FN": int(cm[1,0]), "TP": int(cm[1,1])},
    "signal_discrimination": signal_disc,
    "feature_statistics": {name: {"mean": float(np.mean(features[:,i])),
                                   "std": float(np.std(features[:,i]))}
                           for i, name in enumerate(FEATURE_NAMES)},
}

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2, default=str)


# ===========================================================================
# Final summary
# ===========================================================================

log("\n" + "=" * 70)
log("  FINAL RESULTS")
log("=" * 70)
log(f"  Dataset: {len(df):,} | Train {len(train_idx):,} / Val {len(val_idx):,} / Test {len(test_idx):,}")
log(f"")
log(f"  COMPOSITE SCORE (Phase-6 threshold, 7 signals):")
log(f"    Default (0.50):  BA={default_m['balanced_accuracy']:.4f}  F1={default_m['f1']:.4f}  MCC={default_m['mcc']:.4f}")
log(f"    Optimal ({optimal['threshold']:.2f}):  BA={optimal['balanced_accuracy']:.4f}  FNR={optimal['fnr']:.4f}  FPR={optimal['fpr']:.4f}")
log(f"    AUC-ROC: {auc:.4f}")
log(f"")
log(f"  ML CLASSIFIERS (5-fold CV):")
for name, r in clf_results.items():
    log(f"    {name:25s}: BA={r['balanced_accuracy']:.4f} CI=[{r['balanced_accuracy_ci_95'][0]:.4f}, {r['balanced_accuracy_ci_95'][1]:.4f}] F1={r['f1']:.4f} AUC={r['auc_roc']:.4f}")
log(f"")
log(f"  Best ML: {best_ml} (BA={clf_results[best_ml]['balanced_accuracy']:.4f})")
log(f"  Target (BA >= 0.90): {'MET' if clf_results[best_ml]['balanced_accuracy'] >= 0.90 else 'NOT MET'}")
log(f"")
log(f"  SIGNAL DISCRIMINATION:")
for name, d in sorted(signal_disc.items(), key=lambda x: -x[1]['delta']):
    log(f"    {name:20s}: delta={d['delta']:.4f} [{d['status']}]")
log("=" * 70)
log(f"\n  Report saved to {REPORT_PATH}")
