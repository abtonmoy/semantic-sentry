# SemanticSentry — Library Enhancement: Additional Drift Metrics

Current metrics cover three axes: **CKA** (global structural similarity), **NPS** (local neighborhood retention), **Isotropy Δ** (spectral shape). This document tracks candidate metrics to extend coverage along axes the current set misses.

## Gaps in current coverage

The existing three are good but leave four blind spots:

1. **No magnitude in original units.** CKA is a unitless similarity — it tells you "are these similar?" but not "by how much did embeddings move?"
2. **No directionality on local drift.** NPS collapses two different failure modes (false-neighbors introduced vs true-neighbors lost) into one number.
3. **No distributional view.** None of the current metrics check whether the marginal embedding distribution has shifted (e.g., scale-only drift from quantization).
4. **No non-linear structure check.** Linear CKA can't see kernel-level changes that matter for non-linear downstream heads.

Every metric below is justified against one of these four gaps.

---

## Global structure (sibling to CKA)

### - [ ] Orthogonal Procrustes distance
- **What it measures:** Frobenius norm of `Z1 - R·Z0` after solving for the optimal rotation `R`. Closed form via SVD of `Z0ᵀZ1`.
- **Why we need it:** Fills gap **#1** — magnitude of drift in original units. CKA says "0.87 similar"; Procrustes says "embeddings moved an average of 0.34 units after best alignment."
- **What it adds:** A directly interpretable drift number that domain experts can reason about. Especially useful in calibration / transfer functions because it has the same units as embedding distances downstream tasks operate on.
- **Cost:** O(n·d²) — one SVD of a d×d matrix. Cheap.
- **Notes:** Rotation-invariant by construction. Pairs naturally with CKA: CKA for "structural similarity", Procrustes for "movement magnitude".

### - [ ] RSA / pairwise-distance correlation
- **What it measures:** Spearman correlation between flat upper triangles of pairwise distance matrices in Z0 and Z1.
- **Why we need it:** Catches monotonic structural changes that Linear CKA can miss. Also the standard "second opinion" metric in neuro-ML and interpretability literature — gives credibility.
- **What it adds:** Rank-based, scale-invariant, no kernel choice required. Very intuitive to non-specialists ("are the same pairs still close?").
- **Cost:** O(n²·d) for the distance matrices + O(n² log n) for ranking.
- **Notes:** Use Spearman, not Pearson, so outlier pairs don't dominate.

### - [ ] RBF-kernel CKA
- **What it measures:** Same HSIC normalization as Linear CKA but with Gaussian Gram matrices `K[i,j] = exp(-γ‖xᵢ-xⱼ‖²)`.
- **Why we need it:** Fills gap **#4** — non-linear structural similarity. Two embedding spaces can have identical Linear CKA but very different local geometry that downstream non-linear heads see.
- **What it adds:** Catches drift Linear CKA can't see when the model develops new non-linear separations.
- **Cost:** O(n²) — same as Linear CKA, with one extra exponentiation per Gram entry.
- **Notes:** Bandwidth `γ` should default to the median heuristic (`1 / median(‖xᵢ-xⱼ‖²)`). Add as a hyperparameter on the metric entry.

### - [ ] SVCCA (Singular Vector CCA)
- **What it measures:** Project Z0 and Z1 onto their top-k singular directions, then compute CCA. Returns mean correlation.
- **Why we need it:** Noise-robust similarity. When models have many near-zero-variance directions (common after fine-tuning), raw CKA gets noisy; SVCCA is stable.
- **What it adds:** Better behavior on high-dim spaces with rank deficiency.
- **Cost:** O(n·d² + k³). More expensive than CKA but tractable.
- **Notes:** Default `k` from variance threshold (e.g., top directions covering 99% of variance).

### - [ ] PWCCA (Projection-Weighted CCA)
- **What it measures:** CCA correlations weighted by direction importance (Morcos et al. 2018).
- **Why we need it:** Sharper than plain CCA at detecting drift that affects task-relevant directions, ignoring drift in irrelevant ones.
- **What it adds:** Often correlates better with downstream task degradation than raw CCA.
- **Cost:** Similar to SVCCA.
- **Notes:** Worth adding only after SVCCA — they share most infrastructure.

---

## Local structure (sibling to NPS)

### - [ ] Trustworthiness & Continuity (Venna & Kaski 2001) ⭐ high priority
- **What it measures:** Two numbers, not one:
  - **Trustworthiness:** penalizes points that are k-NN in Z1 but were *not* k-NN in Z0 (false neighbors introduced).
  - **Continuity:** penalizes points that were k-NN in Z0 but are *not* k-NN in Z1 (true neighbors lost).
- **Why we need it:** Fills gap **#2** — directionality on local drift. NPS = 0.6 could mean "retrieval introduces lots of garbage near-misses" (low Trust, high Continuity) or "retrieval misses things it used to find" (high Trust, low Continuity). These need different remediations.
- **What it adds:** Actionable diagnostics. Tells users *what kind* of drift they have, not just *how much*. Most valuable single addition to NPS.
- **Cost:** O(n²) or FAISS-accelerated, same shape as NPS.
- **Notes:** Should register two metrics (`trustworthiness`, `continuity`) and expose them as a pair.

### - [ ] Mutual k-NN consistency
- **What it measures:** Of point `i`'s k-NN in Z0, how many of those neighbors *also* have `i` in their top-k in Z1? (Strict mutual relation, not one-way.)
- **Why we need it:** Filters out hubness artifacts that inflate NPS in high-dim spaces. Embedding spaces are notorious for "hub" points that appear in everyone's nearest-neighbor list.
- **What it adds:** A stricter local-drift score less fooled by hubs.
- **Cost:** Same as NPS.
- **Notes:** Often a better predictor of retrieval failure than vanilla NPS.

### - [ ] Mean rank shift
- **What it measures:** For each anchor pair (i,j), how does j's rank in i's neighbor list change between Z0 and Z1? Mean absolute change.
- **Why we need it:** NPS is binary at the k-boundary — a neighbor at rank-k vs rank-(k+1) flips the count. Mean rank shift is continuous.
- **What it adds:** Smoother metric that doesn't have NPS's cliff edge. Good for plotting drift over many checkpoints.
- **Cost:** O(n²) — need full rank lists, not just top-k.

### - [ ] NPS curve at multiple k
- **What it measures:** NPS evaluated at k ∈ {1, 5, 10, 50, 100}.
- **Why we need it:** Drift can be scale-dependent — local clusters intact (NPS@5 high) but global topology disrupted (NPS@100 low), or vice versa.
- **What it adds:** Almost free to compute (you build the full neighbor list anyway). Reveals which scale of structure the drift affected.
- **Cost:** Sub-linear extra on top of single-k NPS.
- **Notes:** Could just be an extension to the existing NPS metric API rather than a new metric.

---

## Spectral / geometric (sibling to Isotropy Δ)

### - [ ] Effective rank Δ (Roy & Vetterli 2007) ⭐ high priority
- **What it measures:** `exp(H(σ̃²))` where `σ̃² = σ² / Σσ²` is the normalized singular value distribution. Delta between Z0 and Z1.
- **Why we need it:** Current Isotropy Δ uses `σ_min/σ_max`, which is dominated by the noise floor — two very different spaces can have nearly identical ratios because both have at least one tiny singular value.
- **What it adds:** A robust effective-dimensionality measure that uses all singular values, not just the extremes.
- **Cost:** One SVD per snapshot — same as current Isotropy. Can cache.
- **Notes:** You already have `effective_dimensionality()` in `metrics/isotropy.py:81` — just expose it as a registered `(Z0, Z1) → float` metric.

### - [ ] Participation ratio Δ
- **What it measures:** `PR = (Σσ²)² / Σσ⁴`, then delta.
- **Why we need it:** Cross-check on Effective Rank. Same family but different sensitivity to the tail of the spectrum.
- **What it adds:** Cheap sanity-check metric — if PR and Effective Rank disagree, the spectrum has unusual shape worth investigating.
- **Cost:** Negligible after SVD.

### - [ ] Stable rank Δ
- **What it measures:** `‖Z‖_F² / ‖Z‖_2²`, then delta.
- **Why we need it:** Cheapest effective-rank proxy — no full SVD needed, just Frobenius norm and top singular value.
- **What it adds:** Fast effective-rank surrogate for use in production monitoring loops.
- **Cost:** O(n·d) Frobenius + O(n·d) power iteration for top σ.

### - [ ] Spectral entropy Δ
- **What it measures:** Shannon entropy of normalized singular values.
- **Why we need it:** Different normalization than effective rank — can be more sensitive when spectrum has a long tail.
- **What it adds:** Optional companion to effective rank for users who want maximum spectral information.
- **Cost:** Negligible after SVD.
- **Notes:** Lower priority — Effective Rank usually suffices.

---

## Distributional (currently missing axis)

### - [ ] MMD (Maximum Mean Discrepancy, RBF kernel) ⭐ high priority
- **What it measures:** Kernel two-sample test statistic between the embedding distributions of Z0 and Z1.
- **Why we need it:** Fills gap **#3** entirely — currently no metric tells you if the marginal distribution has shifted. Particularly important for catching quantization drift (where points rotate slightly but the cloud shape changes).
- **What it adds:** Detects drift NPS and CKA miss when anchors are correlated. Standard in distribution-shift literature, so users will trust it.
- **Cost:** O(n²·d). For large anchor sets, use random Fourier features for O(n·d·m) approximation.
- **Notes:** Stochastic if RBF bandwidth uses random subsampling — needs a seed. See "API note" below.

### - [ ] Sliced Wasserstein distance
- **What it measures:** Average Wasserstein-1 distance between Z0 and Z1 projected onto L random directions.
- **Why we need it:** Captures shape + location distributional shifts at sub-quadratic cost.
- **What it adds:** Where MMD captures kernel-level differences, Sliced Wasserstein captures geometric transport cost — closer to "how much mass moved how far."
- **Cost:** O(L·n·log n) for L projections (typically L ≈ 50).
- **Notes:** Stochastic — needs a seed.

### - [ ] Fréchet distance (FID-style)
- **What it measures:** Wasserstein-2 between Gaussian fits to Z0 and Z1 — combines mean shift and covariance shift.
- **Why we need it:** Familiar to vision/generative practitioners (FID, KID family). Lowers adoption barrier.
- **What it adds:** A single number combining first and second moments. Strong baseline.
- **Cost:** O(d³) for the covariance matrix square root.
- **Notes:** Sensitive to outliers; not robust on small anchor sets.

### - [ ] Embedding-norm KS test
- **What it measures:** Kolmogorov-Smirnov statistic between the distributions of `‖z‖` in Z0 and Z1.
- **Why we need it:** Catches pure-scale drift that cosine-based metrics miss entirely. This is a *real* failure mode after LoRA / quantization where directions are preserved but norms drift.
- **What it adds:** Cheap, statistically principled detector for a specific failure mode none of the current metrics catch.
- **Cost:** O(n log n).
- **Notes:** Important if users care about un-normalized embeddings (e.g., dot-product retrieval, not just cosine).

---

## Behavioral (downstream-flavored, no labels needed)

### - [ ] Self-retrieval top-1 consistency ⭐ high priority
- **What it measures:** For each anchor, find its top-1 nearest neighbor in Z0 and in Z1. Report fraction of anchors where these match.
- **Why we need it:** Most explainable metric you could ship. Maps directly to retrieval-task degradation: "X% of queries would have returned a different result."
- **What it adds:** The "demo metric" — what you put on a dashboard for non-ML stakeholders. Stronger ROI than any of the abstract similarity metrics.
- **Cost:** O(n²) once, O(1) per anchor after.
- **Notes:** Coarser than NPS but interpretable; consider also exposing top-5 and top-10 variants.

### - [ ] Cosine-rank correlation
- **What it measures:** For each anchor pair (i,j), Spearman correlation between `cos(Z0_i, Z0_j)` and `cos(Z1_i, Z1_j)`.
- **Why we need it:** RSA variant on cosine instead of Euclidean — especially relevant since you already L2-normalize embeddings.
- **What it adds:** A normalized-space analogue of RSA that's natural for retrieval embeddings.
- **Cost:** O(n²).

---

## Recommended priority order

If you're adding metrics in waves, this is the order that maximizes coverage-per-effort:

| Wave | Metric | Gap filled | Rationale |
|------|--------|-----------|-----------|
| 1 | Orthogonal Procrustes distance | #1 magnitude | Closed form, ~30 LOC, fills the biggest gap |
| 1 | Trustworthiness & Continuity | #2 directionality | Single most actionable diagnostic improvement |
| 1 | MMD | #3 distributional | Opens an entire missing axis |
| 1 | Effective Rank Δ | sharper #4 | You already have the building block |
| 1 | Self-retrieval top-1 consistency | explainability | Highest stakeholder ROI |
| 2 | RSA / pairwise-distance correlation | non-linear cross-check | Easy follow-up to Procrustes |
| 2 | RBF-kernel CKA | #4 non-linear | Drop-in extension to existing CKA code |
| 2 | Embedding-norm KS | scale drift | Quick to write, catches a real failure mode |
| 2 | Mutual k-NN consistency | hubness-robust local | Stricter NPS variant |
| 3 | SVCCA / PWCCA | noise-robust global | Higher complexity; only if Wave 1+2 insufficient |
| 3 | Sliced Wasserstein | transport distributional | If MMD proves insufficient |
| 3 | Participation Ratio, Stable Rank, Spectral Entropy | spectral cross-checks | Cheap; bundle with Effective Rank |
| 3 | Mean rank shift, NPS-curve | smoother local | Optional refinements |

---

## API note before implementing

The registry's determinism check (`metrics/registry.py:106-130`) calls the function on random data and rejects any function whose two calls don't return identical floats. This will reject any stochastic metric (MMD with random kernel bandwidth, Sliced Wasserstein with random projections) unless the randomness is seeded.

Two paths to handle this:

- **Closure approach (simpler).** Wrap stochastic metrics in a factory like `make_mmd(seed=0, kernel="rbf", bw="median")` that returns a closed-over function with all randomness baked in. No registry changes needed. Trade-off: hyperparameters are frozen at registration time.

- **Config-on-entry approach (more flexible).** Extend `MetricEntry` with an optional `params: dict` field; have the registry call `fn(Z0, Z1, **entry.params)`. Backward-compatible if `params` defaults to `{}`. Lets calibration profiles record which hyperparameters were used — useful for reproducibility audits across model versions.

**Recommendation:** Start with the closure approach for Wave 1. Move to config-on-entry in Wave 2 once you have multiple metrics that share hyperparameters and need them recorded in `CalibrationProfile`.
