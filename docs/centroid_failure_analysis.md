# Why Centroid-Based Classification Fails with Cosine Similarity

## Key Finding

**Centroid-based scoring using cosine similarity fails for ALL embedding models** when classifying prompt injection attacks. This failure is **methodological**, not architectural — the same embeddings work well with linear probes.

---

## Experimental Results

| Model | Params | Centroid Cos. Sim. | Accuracy | AUC-ROC | Linear Probe AUC |
|-------|--------|-------------------|----------|---------|------------------|
| bge-large-en-v1.5 | 335M | 0.9864 | 55.2% | 0.533 | 0.884 |
| all-MiniLM-L6-v2 | 22M | 0.9912 | 52.8% | 0.512 | 0.844 |
| e5-small-v2 | 33M | 0.9789 | 56.1% | 0.541 | 0.860 |
| bge-small-en-v1.5 | 33M | 0.9845 | 54.3% | 0.528 | 0.855 |

**All models collapse**: centroid similarity > 0.95, AUC-ROC ~0.5 (random).

---

## Mathematical Explanation

### 1. The Centroid Method

Given embeddings $\mathbf{x}_i \in \mathbb{R}^d$ for each class $c \in \{0, 1\}$:

$$\mu_c = \frac{1}{|C|} \sum_{i \in C} \mathbf{x}_i$$

Classification rule (cosine similarity):

$$\hat{y} = \arg\max_c \frac{\mathbf{x} \cdot \mu_c}{\|\mathbf{x}\| \|\mu_c\|}$$

### 2. Why It Fails: Centroid Convergence

When centroids overlap ($\cos(\mu_0, \mu_1) \to 1$):

$$\cos(\mu_0, \mu_1) = \frac{\mu_0 \cdot \mu_1}{\|\mu_0\| \|\mu_1\|} \approx 0.99$$

This means:

$$\mu_0 \approx \mu_1$$

The decision boundary becomes degenerate — both centroids are essentially the same point in high-dimensional space.

### 3. The Geometric Intuition

In $d$-dimensional space, the cosine similarity measures **angular separation**:

$$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$$

When $\cos(\theta) > 0.99$:
- Angular separation $\theta < 8°$
- The two class distributions overlap almost completely along the angular axis
- All queries have nearly identical similarity to both centroids

### 4. Why Embeddings Cause This

Embedding models encode **semantic meaning**, not **adversarial intent**:

$$\mathcal{E}(\text{"Ignore all instructions"}) \approx \mathcal{E}(\text{"Explain photosynthesis"})$$

Both are English imperative sentences addressing an AI assistant. In the embedding space:
- **Semantic similarity** (what it says) dominates
- **Intent** (what it tries to do) is not encoded

### 5. The Dimensionality Curse

In high-dimensional spaces ($d = 1024$ for bge-large):
- Random vectors are nearly orthogonal ($\cos \theta \approx 0$)
- But class centroids average many vectors, reducing variance
- The mean of many similar vectors converges to a point
- Both classes converge to similar points because the signal (intent) is weak

Mathematically:

$$\text{Var}(\mu_c) = \frac{\sigma^2}{n} \to 0 \text{ as } n \to \infty$$

The centroids become point estimates with no variance, regardless of class.

### 6. Why Linear Probes Work

Linear probes find an optimal separating hyperplane:

$$\mathbf{w}^T \mathbf{x} + b = 0$$

This hyperplane can exploit **non-angular structure** in the embedding space:
- Magnitude differences across dimensions
- Non-linear relationships
- Correlations between dimensions

The discriminative information exists but is **not accessible via centroid proximity**.

---

## Why StandardScaler + Euclidean Distance Works Better

### The Experimental Finding

| Method | AUC-ROC | What It Shows |
|--------|---------|---------------|
| Cosine similarity (raw) | ~0.5 | Angle-based: centroids overlap |
| Euclidean + StandardScaler | ~0.75 | Distance-based: separation exists |
| Linear probe | ~0.84 | Optimal boundary: best separation |

StandardScaler + Euclidean distance achieves **AUC ~0.75**, significantly better than cosine similarity (~0.5).

### The Mathematical Transformation

**StandardScaler** transforms each dimension $j$ to have mean 0 and variance 1:

$$x_{ij}^{scaled} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

where $\mu_j$ and $\sigma_j$ are the mean and standard deviation of dimension $j$ across all samples.

**Euclidean distance** in scaled space:

$$d(\mathbf{x}, \mathbf{c}) = \sqrt{\sum_{j=1}^{d} (x_j^{scaled} - c_j^{scaled})^2}$$

### Why This Changes Everything

#### 1. Dimension Rescaling

In the original embedding space:
- Some dimensions have **high variance** (dominate cosine similarity)
- Some dimensions have **low variance** (ignored by cosine similarity)
- Discriminative information may exist in **low-variance dimensions**

After StandardScaler:
- **All dimensions contribute equally** (variance = 1)
- Low-variance dimensions that contain discriminative signals get **upweighted**
- High-variance dimensions that contain noise get **downweighted**

#### 2. The Mathematical Proof

Let $\mathbf{x} \in \mathbb{R}^d$ be an embedding vector. Decompose it into:

$$\mathbf{x} = \mathbf{x}_{high\_var} + \mathbf{x}_{low\_var}$$

where $\mathbf{x}_{high\_var}$ contains dimensions with high variance, and $\mathbf{x}_{low\_var}$ contains dimensions with low variance.

**Cosine similarity** focuses on $\mathbf{x}_{high\_var}$:

$$\cos(\mathbf{x}, \mathbf{c}) = \frac{\mathbf{x} \cdot \mathbf{c}}{\|\mathbf{x}\| \|\mathbf{c}\|} \approx \frac{\mathbf{x}_{high\_var} \cdot \mathbf{c}_{high\_var}}{\|\mathbf{x}_{high\_var}\| \|\mathbf{c}_{high\_var}\|}$$

The discriminative signal in $\mathbf{x}_{low\_var}$ is **drowned out**.

**Euclidean distance after StandardScaler** uses all dimensions equally:

$$d^{scaled}(\mathbf{x}, \mathbf{c}) = \sqrt{\sum_{j=1}^{d} \left(\frac{x_j - \mu_j}{\sigma_j} - \frac{c_j - \mu_j}{\sigma_j}\right)^2}$$

$$= \sqrt{\sum_{j=1}^{d} \frac{(x_j - c_j)^2}{\sigma_j^2}}$$

This is equivalent to a **weighted Euclidean distance** where dimensions with low variance ($\sigma_j$ small) get **higher weight**:

$$w_j = \frac{1}{\sigma_j^2}$$

#### 3. The Geometric Interpretation

**Before scaling:**
- High-variance dimensions dominate the geometry
- Centroids overlap because the dominant dimensions are semantically similar
- The discriminative signal in low-variance dimensions is invisible

**After scaling:**
- Each dimension contributes equally to the distance
- Low-variance dimensions that encode intent differences become visible
- The centroids separate because the scaled space reveals the discriminative structure

#### 4. Why AUC Improves from 0.5 to 0.75

The improvement comes from **accessing previously hidden information**:

$$\text{AUC}_{cosine} \approx 0.5 \quad \text{(only angular information)}$$

$$\text{AUC}_{scaled} \approx 0.75 \quad \text{(angular + magnitude information)}$$

The 0.25 AUC improvement represents the discriminative information that exists in the **magnitude differences across dimensions**, which cosine similarity ignores.

### 5. Why It Still Doesn't Match Linear Probes

| Method | AUC | What It Can Access |
|--------|-----|-------------------|
| Cosine similarity | ~0.5 | Angular separation only |
| Euclidean + StandardScaler | ~0.75 | Angular + magnitude differences |
| Linear probe | ~0.84 | Optimal combination of all dimensions |

Linear probes still outperform because:
1. They learn **dimension-specific weights** (not just equal weighting)
2. They can exploit **correlations between dimensions**
3. They find the **optimal separating hyperplane**, not just nearest centroid

---

## The Core Insight

| Method | What It Measures | Can It Detect Intent? |
|--------|------------------|----------------------|
| Cosine similarity | Angular separation | ❌ No (ignores magnitude) |
| Euclidean + StandardScaler | Scaled distance | ✅ Partially (reveals hidden info) |
| Linear probe | Optimal hyperplane | ✅ Yes (learns optimal boundary) |

**The failure is in the metric, not the embeddings. StandardScaler reveals discriminative information that cosine similarity hides.**

---

## Implications

1. **RAG-based security is flawed**: Similarity-based retrieval cannot distinguish malicious from benign intent
2. **Embedding models need fine-tuning**: General-purpose embeddings don't encode security-relevant features
3. **Supervised classifiers are necessary**: Linear probes or hybrid approaches can learn the decision boundary
4. **Centroid scoring should be abandoned**: For security classification tasks

---

## References

- Paper: "Benchmarking Classification Methods for Prompt Injection Detection"
- Dataset: Guardrailer Dataset v1 (722,842 samples)
- Models: BAAI/bge-large-en-v1.5, all-MiniLM-L6-v2, e5-small-v2, bge-small-en-v1.5
