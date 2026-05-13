# SemanticSentry — Library Enhancement (v2)

Comprehensive metric and infrastructure expansion. Every item below earns its slot for at least one of:

- **[paper]** — supports a §3 / §4 / §7 claim, or helps close §3.3
- **[lib]** — broadens analytical coverage, hardens diagnostics
- **[mlops]** — deployment monitoring, CI/CD gating, alerting
- **[research]** — post-v1.0.0 paper, deferred experiments (§7.3, §7.4)

Experiments to validate any of this against the existing §4 checkpoints live in [`paper_extention.md`](./paper_extention.md). This doc is the *codebase* roadmap; that one is the *experimental* roadmap.

---

## How to read this

- **Families A–F** are metric groups. Each metric has a one-paragraph rationale + tags + cost.
- **Families G–I** are cross-cutting infrastructure (API, anchor protocol, calibration).
- **Phased rollout** at the end maps everything to v0.2 / v0.3 / future-work waves.
- **Coverage matrix** below gives the 30-second summary.

---

## Coverage matrix

| Family | Metrics | Attacks §3.3? | Wave |
|---|---|---|---|
| A — Behavioral / ranking | 8 | Yes — primary attack | v0.2 (most) |
| B — Local structure (NPS family) | 7 | Maybe (B1/B2 asymmetry) | v0.2 (B1/B2/B5), v0.3 (rest) |
| C — Global structure (CKA family) | 6 | No — robustness only | v0.3 |
| D — Spectral | 6 | Maybe (D5) | v0.3 |
| E — Distributional | 5 | No — robustness, quantization | v0.3 / future |
| F — Temporal layer | 5 | Yes — F5 attack | v0.2 (F1/F2/F3), v0.3 (F4/F5) |
| G — Registry API | 5 | — | v0.2 (needed to register A–F) |
| H — Anchor-set protocol | 4 | Yes — H1 enables Family A | v0.2 |
| I — Calibration & severity | 3 | Yes — I1 enables §7.4 fix | v0.3 |

**Total: 36 new metrics + 12 infra items.** Most are 30–200 LOC each. The temporal layer (F) is a meta-feature that wraps any registered metric.

---

# Family A — Behavioral / ranking-stability

The new family. Captures how the model's *scoring behavior* shifts across pair-of-points, not how individual neighborhoods shift. This is the family that has a real shot at closing §3.3, because anti-aligned drift can preserve local neighborhoods while destroying rankings, and aligned drift can reshape local neighborhoods while preserving rankings.

Requires **Q/D anchor partitioning** (see H1) — split the anchor set into a query subset and document subset (default 50/50 with fixed seed).

### - [x] A1. Score-distribution JSD
- **What:** Jensen–Shannon divergence between the distribution of pairwise cosine similarities `S_M(q,d)` in Z0 vs Z1, marginalized over (q,d) pairs.
- **Rationale:** Catches global score-calibration shifts (e.g., the whole score distribution flattens or shifts left under anti-aligned drift). Symmetric, bounded in [0, log 2].
- **Tags:** [paper] [lib] [mlops]
- **Cost:** O(n²·d) for the |Q|×|D| similarity matrix per snapshot; O(n²) for the histogram + JSD.
- **Notes:** Histogram bin count is a hyperparameter (default 100). Robust to label-free use.

### - [x] A2. Mean absolute pointwise score change
- **What:** `(1/|Q||D|) · Σ |S_baseline(q,d) − S_updated(q,d)|`, normalized by baseline scale.
- **Rationale:** Cheaper sibling to JSD, captures pointwise (not just distributional) score drift. Useful when you want a single interpretable number ("scores moved by 0.07 on average").
- **Tags:** [paper] [lib] [mlops]
- **Cost:** O(n²·d) for similarity matrices; O(n²) for the sum.

### - [x] A3. Per-query Rank-Biased Overlap (RBO)
- **What:** For each query q, RBO between baseline and updated ranking of D. Return mean across queries.
- **Rationale:** Top-weighted ranking metric — disagreement at rank 1 costs more than at rank 100. Matches what retrieval users actually care about. Bounded [0, 1].
- **Tags:** [paper] [lib] [mlops]
- **Cost:** O(|Q|·|D|·log|D|) ranking + O(|Q|·|D|) RBO.
- **Notes:** Persistence parameter `p` (typically 0.9) is a hyperparameter. RBO is preferred over Kendall τ because it's top-heavy and handles disjoint tails.

### - [x] A4. Per-query Kendall τ
- **What:** For each query q, Kendall τ between baseline and updated ranking of D. Return mean.
- **Rationale:** Standard rank-correlation; familiar to reviewers. Pairs with RBO — τ captures whole-list correlation, RBO captures top-list overlap.
- **Tags:** [paper] [lib]
- **Cost:** O(|Q|·|D|·log|D|) via Knight's algorithm.
- **Notes:** Less informative than RBO when only top results matter. Worth including for credibility.

### - [x] A5. Self-retrieval top-k consistency
- **What:** For each anchor `i`, find its top-k nearest neighbors in Z0 and in Z1. Report fraction of anchors where the top-k *sets* match exactly (k=1, 5, 10).
- **Rationale:** Most interpretable metric in the suite. Maps directly to retrieval-task degradation: "X% of queries would return a different top-1 result." Strong dashboard candidate.
- **Tags:** [paper] [lib] [mlops]
- **Cost:** O(n²) once; O(1) per anchor after.
- **Notes:** Coarser than NPS but more interpretable for non-ML stakeholders.

### - [ ] A6. Reciprocal rank shift
- **What:** For each anchor pair (i, j) where j was top-ranked for i in Z0, the change in 1/rank of j among i's neighbors in Z1.
- **Rationale:** Smoother than top-k consistency (continuous instead of binary). Catches "j is now rank 5 instead of rank 1" rather than just "j is no longer top-k."
- **Tags:** [paper] [lib]
- **Cost:** O(n²).
- **Notes:** Bounded in [-1, 1] after normalization.

### - [ ] A7. Paraphrase-invariance preservation
- **What:** Given labeled paraphrase pairs `(x, x')` in the anchor set, measure `mean(cos(Z_M(x), Z_M(x')))` for each model. Report ratio updated/baseline.
- **Rationale:** Tests whether *semantic equivalence* is preserved — a more orthogonal signal than score-distribution drift because it's anchored to semantic, not geometric, structure. Fardini Doc 2 §6.2 flags this as the right fallback if pure geometric/behavioral signals fail.
- **Tags:** [paper] [research]
- **Cost:** O(n·d) given pairs.
- **Notes:** Requires labeled paraphrase pairs in the anchor set (see H3). Don't ship until anchor protocol supports it.

### - [ ] A8. Frozen-reference agreement
- **What:** Given a frozen "oracle" reference model M_ref, compute cosine between `Z_M(x)` and `Z_M_ref(x)` for each anchor. Mean across anchors. Track delta.
- **Rationale:** The most orthogonal behavioral signal — uses information from outside the drift pair entirely. Production setting: M_ref = the last known-good checkpoint that passed downstream eval.
- **Tags:** [paper] [mlops] [research]
- **Cost:** One extra forward pass per anchor at registration time; O(n·d) for the comparison.
- **Notes:** Requires API extension to register a reference model on the monitor. Pairs naturally with the CI/CD use case.

---

# Family B — Local structure (NPS family)

Extends NPS along the local-neighborhood axis. B1+B2 are the asymmetric decomposition that could discriminate regimes; the rest are robustness improvements.

### - [x] B1. Trustworthiness (priority)
- **What:** For each point, penalize neighbors in Z1's top-k that were *not* in Z0's top-k (false neighbors introduced).
- **Rationale:** Half of the directional NPS decomposition. Anti-aligned drift may introduce more false neighbors (retrieval picks up garbage); aligned drift may not. Falsifiable test for §3.3 attack #3.
- **Tags:** [paper] [lib] [mlops]
- **Cost:** Same as NPS — O(n²) or FAISS-accelerated.

### - [x] B2. Continuity (priority)
- **What:** For each point, penalize neighbors in Z0's top-k that are *not* in Z1's top-k (true neighbors lost).
- **Rationale:** Other half of the directional decomposition. Aligned drift may lose more true neighbors (model has *intentionally* reshaped); anti-aligned may not.
- **Tags:** [paper] [lib] [mlops]
- **Cost:** Same as NPS.
- **Notes:** Register B1 and B2 as paired metrics. The *ratio* T/C is the discriminating signal.

### - [ ] B3. Mutual k-NN consistency
- **What:** For each point i, fraction of i's k-NN in Z0 that have i in *their* top-k in Z1 (strict mutual relation).
- **Rationale:** Filters out hubness artifacts — single "hub" points appearing in everyone's neighborhood list don't dominate the score.
- **Tags:** [lib] [mlops]
- **Cost:** O(n²) or FAISS.

### - [ ] B4. Mean rank shift
- **What:** For each anchor pair (i, j), j's rank-in-i's-neighbors changes by Δ. Mean |Δ| across all pairs.
- **Rationale:** Smoother than NPS — no cliff at the k-boundary. Useful for time-series plots over many checkpoints.
- **Tags:** [lib]
- **Cost:** O(n²) for full rank lists.

### - [x] B5. NPS curve (multi-k)
- **What:** NPS evaluated at k ∈ {1, 5, 10, 25, 50, 100}.
- **Rationale:** Drift can be scale-dependent. The §6.2 sanity check shows 4.6 pp spread across k — that's signal, not noise. Reveals which scale of local structure the drift affected.
- **Tags:** [paper] [lib]
- **Cost:** ~free given the existing FAISS index.
- **Notes:** API: register `nps_at_k` factory that returns one MetricEntry per k.

### - [ ] B6. LCMC (Local Continuity Meta-Criterion)
- **What:** k-NN overlap normalized by expected random overlap: `(NPS·k − k²/(n−1)) · 1/(1 − k/(n−1))`.
- **Rationale:** Subtracts the random baseline NPS already shows in the §6.2 sanity check (`k/(n−1)`). Gives a "skill above random" interpretation in [0, 1].
- **Tags:** [lib]
- **Cost:** Trivial post-processing of NPS.

### - [ ] B7. Hubness shift
- **What:** Skewness of the distribution of "k-occurrences" (how many times each point appears in others' top-k). Track delta between Z0 and Z1.
- **Rationale:** Hubness is a documented embedding pathology — drift can introduce or remove hub points, which silently breaks retrieval. Doesn't show up in NPS.
- **Tags:** [lib] [mlops] [research]
- **Cost:** O(n·k) post-processing of the FAISS index.

---

# Family C — Global structure (CKA family)

Robustness expansions to CKA. None of these attack §3.3 — they're library breadth.

### - [ ] C1. Orthogonal Procrustes distance
- **What:** Frobenius norm of `Z1 − R·Z0` after solving for optimal rotation R via SVD of `Z0ᵀZ1`.
- **Rationale:** Rotation-invariant magnitude of drift in original units. CKA tells you "how similar?", Procrustes tells you "by how much did things move?" Useful in transfer functions because the units match downstream-task distances.
- **Tags:** [lib]
- **Cost:** O(n·d² + d³) for the SVD.

### - [ ] C2. RSA / pairwise-distance correlation
- **What:** Spearman correlation between flat upper triangles of pairwise distance matrices in Z0 and Z1.
- **Rationale:** Standard "second opinion" metric from interpretability literature. Rank-based, scale-invariant. Intuitive to non-specialists.
- **Tags:** [lib]
- **Cost:** O(n²·d) for distance matrices, O(n² log n) for ranking.

### - [ ] C3. RBF-kernel CKA
- **What:** CKA with Gaussian Gram matrices `K[i,j] = exp(−γ‖xᵢ−xⱼ‖²)` instead of linear.
- **Rationale:** Catches non-linear structural similarity Linear CKA misses. Default `γ` via median heuristic.
- **Tags:** [lib]
- **Cost:** O(n²) with extra exponentiation.

### - [ ] C4. SVCCA
- **What:** Project Z0, Z1 onto top-k singular directions, then CCA. Mean correlation.
- **Rationale:** Noise-robust similarity. Better than CKA when models have near-zero-variance directions.
- **Tags:** [lib] [research]
- **Cost:** O(n·d² + k³).

### - [ ] C5. PWCCA
- **What:** CCA correlations weighted by direction importance (Morcos et al. 2018).
- **Rationale:** Sharper than SVCCA at detecting drift in task-relevant directions.
- **Tags:** [lib] [research]
- **Cost:** Similar to SVCCA.

### - [ ] C6. Distance correlation (dCor)
- **What:** Energy-based dependence measure between Z0 and Z1; non-linear, no kernel choice.
- **Rationale:** Tests dependence (not just linear similarity) without picking a kernel bandwidth. Complementary to RBF-CKA.
- **Tags:** [lib]
- **Cost:** O(n²·d).

---

# Family D — Spectral

Replaces / extends the current `isotropy_delta` (which uses fragile `σ_min/σ_max`). D5 is the only one with a §3.3 angle.

### - [ ] D1. Effective rank Δ (priority)
- **What:** `exp(H(σ̃²))` where `σ̃² = σ² / Σσ²`. Delta between snapshots.
- **Rationale:** Robust effective-dimensionality from full spectrum, not just extremes. You already have `effective_dimensionality()` in `metrics/isotropy.py:81` — just expose as a registered `(Z0,Z1)→float`.
- **Tags:** [lib] [mlops]
- **Cost:** One SVD per snapshot (cacheable).

### - [ ] D2. Participation ratio Δ
- **What:** `PR = (Σσ²)² / Σσ⁴`, then delta.
- **Rationale:** Cross-check on effective rank — different tail sensitivity. If PR and effective rank disagree, the spectrum has an unusual shape worth investigating.
- **Tags:** [lib]
- **Cost:** Negligible after SVD.

### - [ ] D3. Stable rank Δ
- **What:** `‖Z‖_F² / ‖Z‖_2²`, then delta.
- **Rationale:** Cheapest effective-rank proxy — no full SVD, just Frobenius + power iteration for top σ. Good for tight production monitoring loops.
- **Tags:** [lib] [mlops]
- **Cost:** O(n·d) Frobenius + O(n·d) power iteration.

### - [ ] D4. Spectral entropy Δ
- **What:** Shannon entropy of `σ² / Σσ²`, then delta.
- **Rationale:** Different normalization than effective rank; can be more sensitive when spectrum has a long tail.
- **Tags:** [lib]
- **Cost:** Negligible after SVD.

### - [ ] D5. Anisotropy direction shift- **What:** Angle between top singular vector of `Z0_centered` and `Z1_centered`. Range [0, π/2].
- **Rationale:** Captures *where* the dominant embedding direction has rotated, not just whether the spectrum changed magnitude. If anti-aligned drift rotates the dominant direction more than aligned drift, this discriminates.
- **Tags:** [paper] [lib] [research]
- **Cost:** Negligible after top-1 SVD.

### - [ ] D6. IsoScore Δ
- **What:** IsoScore (Rudman et al. 2022) — a more robust isotropy estimator than `σ_min/σ_max`.
- **Rationale:** Modern replacement for the current isotropy metric. Reviewers familiar with embedding literature will expect this.
- **Tags:** [lib]
- **Cost:** O(d²) post-SVD.

---

# Family E — Distributional

Tests whether the marginal embedding distribution shifted. None attacks §3.3 directly (same orthogonality limit as all pure-geometric metrics), but several are valuable for production monitoring and the deferred quantization study (§7.3).

### - [ ] E1. MMD (RBF kernel)
- **What:** Maximum Mean Discrepancy between the embedding distributions of Z0 and Z1 with Gaussian kernel.
- **Rationale:** Kernel two-sample test from distribution-shift literature. Standard, reviewer-trusted. Catches drift NPS / CKA miss when anchors are correlated.
- **Tags:** [lib] [mlops] [research]
- **Cost:** O(n²·d). Use random Fourier features for O(n·d·m) approximation on large anchor sets.
- **Notes:** Stochastic — needs seeded bandwidth (see G3).

### - [ ] E2. Sliced Wasserstein distance
- **What:** Average Wasserstein-1 over L random 1-D projections.
- **Rationale:** Transport-based distance — captures shape + location shifts. Cheaper than full Wasserstein, scalable to high dim.
- **Tags:** [lib] [research]
- **Cost:** O(L·n log n) for L projections.
- **Notes:** Stochastic — needs seed (G3). Default L=50.

### - [ ] E3. Fréchet distance (FID-style)
- **What:** Wasserstein-2 between Gaussian fits to Z0 and Z1.
- **Rationale:** Familiar to vision practitioners (FID). Single number combining mean and covariance shift.
- **Tags:** [lib] [research]
- **Cost:** O(d³) for covariance matrix square root.
- **Notes:** Sensitive to outliers; not robust on small anchor sets.

### - [ ] E4. Embedding-norm KS test- **What:** Kolmogorov–Smirnov statistic between distributions of `‖z‖` in Z0 and Z1.
- **Rationale:** Catches **pure-scale drift** that cosine-based metrics miss entirely. Critical for the deferred quantization study (§7.3) — quantization rotates directions slightly but mainly changes norms.
- **Tags:** [lib] [mlops] [research]
- **Cost:** O(n log n). Cheap.

### - [ ] E5. Pairwise distance distribution KS/JSD
- **What:** KS / JSD between the full distribution of pairwise distances in Z0 vs Z1.
- **Rationale:** Stronger than mean-only distance shift — catches changes in distance variance, multimodality, tail behavior.
- **Tags:** [lib] [research]
- **Cost:** O(n²·d) for distance matrices.

---

# Family F — Temporal layer (meta-feature)

Not metrics themselves — they *wrap* any registered metric to produce time-series signals. From Fardini Doc 1. F3 / F5 are the operational and paper-relevant pieces.

### - [x] F1. Velocity wrapper
- **What:** Given a sequence of snapshots and a metric M, compute `dM/dt` via central differences normalized for unequal checkpoint spacing.
- **Rationale:** Reveals when geometry is moving fastest. The §4.3.1 trajectory peaks at epochs 1→3 — invisible to value-only NPS.
- **Tags:** [paper] [lib] [mlops]
- **Cost:** O(T) per trajectory, where T = number of checkpoints.
- **Notes:** API: `Velocity(metric_name)` returns a function `(snapshots, times) → np.ndarray`.

### - [x] F2. Acceleration wrapper
- **What:** Second finite difference, `d²M/dt²`.
- **Rationale:** Detects deceleration toward plateau, re-acceleration, oscillation.
- **Tags:** [paper] [lib]
- **Cost:** O(T) post-velocity.

### - [x] F3. Plateau detector (priority)
- **What:** Boolean signal: `|velocity| < ε AND |acceleration| < δ for k consecutive checkpoints`.
- **Rationale:** **Highest-ROI library addition.** The §4.3.1 data shows the MLM run is essentially settled by epoch 10 — 97% of geometric change at 20% of compute. Production training loops can save real money. Fardini Doc 1 §4.1.
- **Tags:** [paper] [lib] [mlops]
- **Cost:** O(T) trivially.
- **Notes:** Default thresholds: ε = 0.005, δ = 0.001, k = 3. Tunable against multi-seed noise floors (§7.5).

### - [ ] F4. Decay-residual anomaly score
- **What:** Fit `log|velocity|` to a line in the post-warmup phase. Per-checkpoint anomaly score = residual / residual_std.
- **Rationale:** Flags training-dynamics anomalies — oscillation, re-acceleration, persistent over/undershoot. Needs the §4.6 anti-InfoNCE trajectory as positive control.
- **Tags:** [paper] [lib] [research]
- **Cost:** O(T) per trajectory.
- **Notes:** Honest caveats from Fardini Doc 1 §3.2 — exponential baseline only holds post-warmup, two-phase decay common. Ship as `experimental` flag in v0.3.

### - [ ] F5. Decay-shape regime classifier
- **What:** Classify a trajectory by the *shape* of its `log|velocity|` curve (linear with negative slope? piecewise? oscillating?).
- **Rationale:** **§3.3 attack #2.** If aligned / random / anti-aligned regimes have qualitatively different decay shapes (not just different rates), this is a label-free regime classifier from temporal dynamics alone.
- **Tags:** [paper] [research]
- **Cost:** O(T) per trajectory; classifier fitting is small.
- **Notes:** See `paper_extention.md` Experiment 2 for validation protocol.

---

# Family G — Registry / API changes

The current `MetricRegistry` (`metrics/registry.py`) needs four upgrades to support Families A–F cleanly.

### - [x] G1. Hyperparameter dict on `MetricEntry`
- **What:** Add optional `params: dict` field. Registry calls `fn(Z0, Z1, **entry.params)`.
- **Why:** Families A (Q/D split, RBO persistence), C (CKA bandwidth), E (MMD kernel, SW projections) all need configurable hyperparameters that should be recorded in calibration profiles for reproducibility.
- **API impact:** Backward-compatible — defaults to `{}`. `register(name, fn, params={...})`.
- **Tags:** [lib]

### - [x] G2. Multi-k registration helper
- **What:** `register_at_k(base_name, fn, ks=[1, 5, 10, 25, 50])` — registers one metric per k value.
- **Why:** B5 (NPS curve), A5 (top-k consistency), and several Family B metrics need this idiom.
- **Tags:** [lib]

### - [ ] G3. Stochastic metric seeding pattern
- **What:** Document and enforce: stochastic metrics either (a) close over a fixed seed at registration, or (b) accept a `seed` in `params` (G1). The current `_validate_determinism` check rejects anything random.
- **Why:** Family E (MMD, SW), Family A (any randomized projection variants), Family F4 (regression bootstrap CIs).
- **Tags:** [lib]

### - [x] G4. Temporal wrapper for any base metric
- **What:** `register_with_temporal(name, fn)` registers `{name, name_velocity, name_acceleration}` automatically. The temporal versions accept `list[Snapshot]` + `list[float]` (times) and return per-checkpoint arrays.
- **Why:** Family F is a meta-feature; this exposes it as first-class API rather than a special case in user code.
- **Tags:** [lib] [paper]

### - [ ] G5. Per-tower metric routing (clarify existing)
- **What:** Document and harden the existing per-tower metric flow in `core/monitor.py:172-181`. Currently per-tower metrics use the same registry — should support tower-specific metrics (e.g., behavioral metrics on text tower only, even for multi-tower models).
- **Why:** CLIP-style multi-tower with asymmetric metric selection.
- **Tags:** [lib]

---

# Family H — Anchor-set protocol

Family A requires Q/D partitioning. Family A7 requires labeled paraphrase pairs. §5.2 establishes that anchor-set distribution matters and current handling is implicit. Make all this explicit.

### - [x] H1. Q/D partitioning on `AnchorSet` (priority)
- **What:** Add `partition(ratio=0.5, seed=0) → (AnchorSet_Q, AnchorSet_D)` method. Result is reproducible (seeded), hashed into `version_hash` so paired snapshots can verify they used the same split.
- **Why:** All of Family A depends on this. Without it, behavioral metrics can't be computed.
- **Tags:** [paper] [lib]

### - [x] H2. Anchor-distribution provenance tagging
- **What:** Add `distribution_tag: str` field to `AnchorSet` (e.g., `"training-dist"` / `"OOD"` / `"deployment-prod"`). `Comparison` carries this through.
- **Why:** §5.2 demonstrated 2.3× NPS magnitude difference between training-dist and OOD anchors on the same checkpoint. Without provenance, cross-experiment comparisons silently misalign.
- **Tags:** [lib] [mlops]

### - [ ] H3. Paraphrase / labeled-pair anchor extension
- **What:** Subclass `LabeledPairAnchorSet(AnchorSet)` with `pairs: list[tuple[int, int]]` indicating semantic-equivalence pairs.
- **Why:** Family A7 (paraphrase invariance). More orthogonal to geometry than score-distribution drift.
- **Tags:** [paper] [research]

### - [ ] H4. Multi-anchor evaluation
- **What:** `monitor.compare_across_anchors(s0_list, s1_list)` — runs metrics against multiple anchor sets and returns disaggregated results.
- **Why:** §5.2 recommends consistent anchor protocol *within* an experiment, but cross-anchor comparison is itself a diagnostic (large variance across anchors = anchor-dependent drift, small variance = global drift).
- **Tags:** [lib] [mlops]

---

# Family I — Calibration & severity

The current severity scale uses fixed thresholds (`comparison.py:42-49`); §7.5 explicitly lists "recalibrate severity thresholds against multi-seed noise floors" as an action item. The §7.4 NPS-bound replacement also lives here.

### - [ ] I1. Per-regime transfer functions
- **What:** Extend `LinearTransfer` to accept an optional regime label. Fit one transfer function per regime. At inference, requires either a known regime or a regime classifier (Family A or F5 output).
- **Why:** **Resolves §7.4 open question.** The current `degradation ≥ (1−NPS)` framing is broken — strictly violated under aligned drift, meaningless under anti-aligned. Per-regime transfer functions replace it with an empirical regime-dependent relationship.
- **Tags:** [paper]
- **Notes:** See `paper_extention.md` Experiment 6 for the calibration protocol.

### - [x] I2. Multi-seed noise-floor severity calibration
- **What:** Replace hard-coded severity thresholds with `calibrate_thresholds(reference_snapshots: list[Snapshot])` that derives LOW/MEDIUM/HIGH/CRITICAL boundaries from the std of multi-seed reference runs.
- **Why:** §7.5 action item. The current `NPS > 0.95 = LOW` threshold doesn't reflect that CLIP has σ ≈ 0.018 across seeds while E5 has σ ≈ 0.002 — same threshold means different things for different model families.
- **Tags:** [lib] [mlops]

### - [ ] I3. Regime-classifier severity tier
- **What:** Add a `regime: AlertRegime` field to `Comparison` set by a registered regime classifier (Family A or F5). New severity logic: BLOCK if regime=anti-aligned AND any metric crosses MEDIUM; MONITOR if regime=random; PASS if regime=aligned regardless of magnitude.
- **Why:** Production deployment of the §3.3 attack. Once regime classification works (Exp 1 or Exp 2 in `paper_extention.md`), this is the production-facing API for it.
- **Tags:** [paper] [mlops]

---

# Phased rollout

## v0.2 — Paper-supporting + immediate library wins

**Goal:** ship everything needed for `paper_extention.md` experiments + the high-ROI MLOps additions.

- [x] **G1** — Hyperparameter dict (prerequisite for A, C, E)
- [x] **G2** — Multi-k registration helper
- [x] **H1** — Q/D partitioning (prerequisite for Family A)
- [x] **H2** — Anchor-distribution provenance tagging
- [x] **A1** — Score-distribution JSD
- [x] **A2** — Mean abs pointwise score Δ
- [x] **A3** — Per-query RBO
- [x] **A4** — Per-query Kendall τ
- [x] **A5** — Self-retrieval top-k consistency
- [x] **B1** — Trustworthiness
- [x] **B2** — Continuity
- [x] **B5** — NPS curve (multi-k)
- [x] **F1** — Velocity wrapper
- [x] **F2** — Acceleration wrapper
- [x] **F3** — Plateau detector
- [x] **G4** — Temporal wrapper for any base metric
- [x] **I2** — Multi-seed noise-floor severity calibration

That's 17 items. Each ~30–200 LOC. Realistic for v0.2.

**v0.2 status: complete.** All 17 items landed in `src/semantic_sentry/` with 67 new
unit tests (194/194 unit suite passing, `ruff` clean). Highlights:

- Behavioral metrics (A1–A5) live in a separate `BehavioralMetricRegistry`
  (`src/semantic_sentry/metrics/behavioral.py`) — their `(Z0_Q, Z1_Q, D0, D1)`
  signature can't share the regular registry's `(Z0, Z1)` contract.
- Temporal layer (F1–F3, G4) lives in `src/semantic_sentry/metrics/temporal.py`
  as a module-level dict, also for signature reasons (`(snapshots, times, name)`).
- `AnchorSet.partition(seed)` uses composite version hashes
  `{parent_hash}:{role}:{seed}` so paired Q/D snapshots cross-verify in
  `DriftMonitor.compare` (mismatched role or seed raises).
- I2 severity calibration: `calibrate_thresholds(reference_snapshots)` returns
  a `SeverityCalibration` accepted by `DriftMonitor.compare(calibration=...)`
  via the existing `Comparison.thresholds` override path.
- Plus `ConsoleLogger` is now exported from `semantic_sentry.integrations`
  (was already implemented but unreachable).

## v0.3 — Library robustness

**Goal:** broaden coverage for non-paper use cases. PyPI release candidate.

- [ ] **A6** — Reciprocal rank shift
- [ ] **B3** — Mutual k-NN consistency
- [ ] **B4** — Mean rank shift
- [ ] **B6** — LCMC
- [ ] **B7** — Hubness shift
- [ ] **C1** — Procrustes distance
- [ ] **C2** — RSA
- [ ] **C3** — RBF-kernel CKA
- [ ] **D1** — Effective rank Δ
- [ ] **D2** — Participation ratio Δ
- [ ] **D3** — Stable rank Δ
- [ ] **D5** — Anisotropy direction shift
- [ ] **E4** — Embedding-norm KS
- [ ] **F4** — Decay-residual anomaly (experimental flag)
- [ ] **F5** — Decay-shape regime classifier
- [ ] **G3** — Stochastic metric seeding pattern
- [ ] **G5** — Per-tower metric routing
- [ ] **I1** — Per-regime transfer functions
- [ ] **I3** — Regime-classifier severity tier

19 items.

## Future / research / post-paper

- [ ] **A7** — Paraphrase-invariance preservation (needs H3)
- [ ] **A8** — Frozen-reference agreement
- [ ] **C4** — SVCCA
- [ ] **C5** — PWCCA
- [ ] **C6** — Distance correlation
- [ ] **D4** — Spectral entropy Δ
- [ ] **D6** — IsoScore Δ
- [ ] **E1** — MMD
- [ ] **E2** — Sliced Wasserstein
- [ ] **E3** — Fréchet distance
- [ ] **E5** — Pairwise distance distribution KS/JSD
- [ ] **H3** — Paraphrase / labeled-pair anchor extension
- [ ] **H4** — Multi-anchor evaluation

13 items. Ship as user requests demand them.

---

# Implementation notes

## Stochastic metric pattern

The current `_validate_determinism` check (`metrics/registry.py:106-130`) calls a metric on random data twice and rejects any function whose output differs. This breaks for Families E (MMD, SW with random projections) and any RFF-approximated A1/A2.

Two compatible patterns:

**Closure approach (default for v0.2):** Wrap stochastic metrics in factories that close over a fixed seed at registration.

```python
def make_mmd(seed: int = 0, kernel: str = "rbf", bandwidth: str = "median"):
    rng = np.random.default_rng(seed)
    def mmd(Z0, Z1):
        # use rng only at fit time
        ...
    return mmd

registry.register("mmd_rbf", make_mmd(seed=0), description="...")
```

**Config-on-entry approach (v0.3+):** Use G1's `params` dict to inject seeds at call time, recorded in `CalibrationProfile`.

```python
registry.register("mmd", mmd_fn, params={"seed": 0, "kernel": "rbf"})
# Reproducible across runs because params is part of the calibration profile.
```

The config approach is required for I1's per-regime transfer functions because reproducibility audits across model versions need to know which hyperparameters were used.

## Backward compatibility

The current public API (`DriftMonitor`, `Snapshot`, `Comparison`, `AnchorSet`) does not need breaking changes. All Family A–F additions are new metrics behind the existing registry. Family G–I are additive — new optional fields on existing dataclasses, new methods.

Only breaking change required: **Severity recomputation under I2.** Custom code that relies on the literal threshold values in `comparison.py:42-49` would need updating. Existing comparisons saved to disk are fine — severity is recomputed on load.

## Cross-cutting effort: integrations layer

The §7.5 PyPI release item depends on the integrations layer actually working. Current state (`integrations/__init__.py` doesn't export `ConsoleLogger`, no `WandbLogger` / `MLflowLogger` implementation despite pyproject extras). Either:

1. **Ship ConsoleLogger only**, mark W&B / MLflow as `# TODO` in README, deprioritize the "MLOps ready" tagline until they exist.
2. **Implement minimal W&B / MLflow loggers** (~150 LOC each) that flatten `Comparison` and `ClassificationResult` into the respective logging APIs.

Recommend (1) for v0.2, (2) for v0.3.
