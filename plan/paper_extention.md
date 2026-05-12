# SemanticSentry v1.0.0 — Paper Extensions

Pre-registerable experiments on **existing §4 checkpoints**, targeting the paper's central admitted limitation (§3.3) plus complementary contributions to MLOps and the §7.4 open questions.

Every experiment below uses only checkpoints already produced for §4.1–§4.6. Zero new training runs required for the core proposals (Experiments 1–6). Total marginal compute: single-digit GPU-hours.

The library features each experiment depends on are tracked in [`lib_enhancement.md`](./lib_enhancement.md) — references like "Family A1" point there.

---

## How to read this

- **Experiments 1–3 attack §3.3** — the paper's stated central limitation
- **Experiments 4–5 strengthen the mechanism + MLOps story** — Fardini Doc 1
- **Experiment 6 resolves §7.4** — replaces the broken NPS-bound framing
- **Experiments 7–8 extend §5 methodology contributions** — anchor sensitivity, probe configuration
- Each experiment carries: pre-registration template, falsification criteria, paper-integration target, compute cost

---

## Experiment inventory

| # | Experiment | Attacks | New compute | Decisive outcome? | Paper section |
|---|---|---|---|---|---|
| 1 | Joint-trajectory regime classifier | §3.3 | ~3 GPU-hrs | Yes (B/G ratio separation) | New §3.5 |
| 2 | Decay-shape regime classifier | §3.3 | <1 CPU-hr | Yes (shape-vs-rate ANOVA) | New §3.6 |
| 3 | Trustworthiness/Continuity asymmetry | §3.3 | <1 GPU-hr | Yes (T/C ratio per regime) | §4.3 augment |
| 4 | Plateau-detection validation (cross-regime) | MLOps claim | <1 CPU-hr | Yes (consistency criterion) | §4 augment, §7.5 |
| 5 | Anomaly detection w/ anti-InfoNCE control | Doc 1 §4.2 | <1 CPU-hr | Yes (anti-InfoNCE flagged) | §4.6 augment |
| 6 | Per-regime transfer functions (NPS-bound replacement) | §7.4 | <1 GPU-hr | Yes (per-regime R²) | §2 calibration update |
| 7 | Anchor-set sensitivity sweep | §5.2 extension | ~2 GPU-hrs | Yes (variance bound) | §5.2 augment |
| 8 | Probe-methodology second-task replication | §5.1 extension | ~2 GPU-hrs | Yes (gap reproduces) | §5.1 augment |

---

# Experiment 1 — Joint-trajectory regime classifier

**Source:** Fardini Doc 2. **Primary attack on §3.3.** **Highest leverage.**

The paper §3.3 admits: *"A label-free framework cannot in general predict which regime a particular drift event belongs to from NPS alone."* This experiment tests whether pairing NPS with a label-free behavioral signal (score-distribution drift) closes that gap.

## Hypothesis

At matched geometric drift (matched NPS), the **ratio of behavioral drift to geometric drift** (B/G ratio) separates aligned, random, and anti-aligned regimes by a factor of ≥ 3× at epoch 50 across all three model families.

## Setup

Use existing checkpoints across all three regimes:

| Regime | Checkpoint source | Epochs | Seeds |
|---|---|---|---|
| Aligned | §4.1 contrastive LoRA on E5, CLIP, BERT | 0, 1, 3, 5, 10, 20, 50 | 3 each |
| Random | §4.2 Gaussian σ ∈ {0.04, 0.08, 0.16} | endpoint only | 3 |
| Anti-aligned (MLM) | §4.3.1 + §4.3.2 | 0, 1, 3, 5, 10, 20, 50 | 3 each corpus |
| Anti-aligned (anti-InfoNCE) | §4.6 | 0, 1, 3, 5, 10, 20, 50 | 3 |

## Procedure

1. **Anchor partitioning.** Use the existing OOD anchor set (MS MARCO dev for E5/BERT, MSCOCO val for CLIP). Partition 50/50 into Q and D with seed=42 (recorded). Lib dep: **H1**.
2. **Compute behavioral signal** per checkpoint. Three variants (registered in lib via **A1/A2/A3**):
   - JSD between marginal cosine-similarity distributions S_baseline(q,d) vs S_updated(q,d)
   - Mean absolute pointwise score change, normalized to baseline scale
   - Mean per-query RBO (p=0.9) between baseline and updated rankings of D
3. **Compute geometric signal** — already exists: NPS at the same checkpoints with the same anchor protocol.
4. **Compute B/G ratio** per checkpoint per regime per seed.
5. **Compare ratios across regimes.** Within-regime mean ± std across seeds and families; between-regime separation test.

## Pre-registration

**H_A (joint trajectory works).** The B/G ratio at epoch 50 separates the three regimes by ≥ 3× across all three model families. The behavioral signal correlates with the labeled benchmark trajectory at Pearson r ≥ 0.85 within each regime.

**H_B (partial success).** The ratio separates aligned from anti-aligned cleanly (≥ 3×), but random falls between them at ratio ≈ 1.0–1.5 instead of cleanly intermediate. Still resolves §3.3 but requires a piecewise classifier rather than a continuous score.

**H_C (joint trajectory fails).** Within-regime correlation between behavioral signal and labeled benchmark is r < 0.5 in any regime. The behavioral signal is dominated by the same magnitude information NPS already carries. Negative result.

**Falsification criterion (commit before running):** If within-regime r < 0.5 in any of {MS MARCO, NFCorpus} for any regime, conclude the behavioral signal is not a useful proxy for labeled-benchmark behavior, and report H_C as the result.

## Predicted outcome (from existing §4.3.1 data)

The §4.3.1 trajectory already shows the signature at epoch 3: NPS = 0.59 (matches aligned-regime endpoint 0.60), but MS MARCO has already dropped −4.5 pp (CI-disjoint with aligned's in-CI). The behavioral signal must be carrying something NPS doesn't — the only question is how robustly it separates.

Best estimate: H_A confirmed for aligned vs anti-aligned (high confidence based on §4.3.1 + §4.1 comparison). Random regime placement is uncertain — possibly H_B.

## Compute

~3 GPU-hours. For an OOD anchor of n ≈ 5,000 (§5.2 typical), |Q|×|D| = 6.25 M cosine entries per checkpoint. At 768-dim FP32 ≈ 5 GB; in FP16 ≈ 1.25 GB. Across ~80 checkpoints in §4: under 1 GPU-hour for similarity matrices + ~2 hours for the bootstrap CIs.

## Paper integration

- **New §3.5 "Joint-trajectory regime classification"** — presents the behavioral signal, predicted ratio signatures, trajectory dynamics.
- **§4.3 table augmentation** — add behavioral-signal column to Tables 8, 9, 11.
- **§3.3 revision** — convert admitted limitation into resolved contribution.
- **§7.2 promotion** — make this the highest-priority pre-submission experiment.
- **§2 severity scale** — if H_A confirmed, regime classifier feeds the new severity tier (lib **I3**).

## Honest risks

The behavioral signal is not fully orthogonal to geometry — both derive from the same embeddings. The §3.3 outcome differential at matched NPS (in-CI vs −10 pp vs −65 pp) proves *some* additional signal exists in the embeddings; this experiment tests whether score-distribution drift captures it. If H_C wins, the next step is **paraphrase invariance** or **frozen-reference agreement** (Family A7/A8), which require labeled-pair anchor sets (H3) — out of scope for this submission.

---

# Experiment 2 — Decay-shape regime classifier

**Source:** Fardini Doc 1 §5. **Independent attack on §3.3 via temporal axis.**

The same regime separation tested in Experiment 1, but using the *shape* of log|velocity| trajectories instead of behavioral drift.

## Hypothesis

In the post-warmup phase, log|velocity| of NPS (and other geometric metrics) is approximately linear in training epoch within each regime, but with qualitatively different shapes across regimes — sufficient to classify a trajectory by shape alone.

## Setup

Same checkpoint inventory as Experiment 1, restricted to multi-checkpoint trajectories (excludes §4.2 random regime which has endpoint-only data):

| Regime | Trajectory | Checkpoints |
|---|---|---|
| Aligned | §4.1 contrastive LoRA on E5, CLIP, BERT | 7 × 3 families × 3 seeds |
| Anti-aligned (MLM) | §4.3.1 + §4.3.2 | 7 × 2 corpora × 3 seeds |
| Anti-aligned (anti-InfoNCE) | §4.6 | 7 × 3 seeds |

## Procedure

1. For each trajectory, compute NPS, CKA, isotropy_delta at every checkpoint. Lib dep: **F1**.
2. Compute velocity via central finite differences, normalized for unequal checkpoint spacing.
3. Compute acceleration as second finite difference.
4. Plot log|velocity| vs epoch. Visually inspect for linearity in epochs ≥ 3 (post-warmup; the §4.3.1 trajectory shows epoch 0→1→3 is the transient).
5. Fit least-squares line to post-warmup log|velocity|. Report slope (rate constant λ), intercept, residual std.
6. **Shape vs rate ANOVA**: test whether residual structure (not just slope) differs across regimes.
7. **Classifier head**: fit a small classifier (logistic regression on shape features: slope, intercept, residual structure, R²) and report cross-validated regime-classification accuracy.

## Pre-registration

**H_1 (post-warmup linearity).** Log|velocity| vs epoch in the post-warmup phase (defined as epoch ≥ 3) is fit by a line with R² ≥ 0.9 across all aligned and anti-aligned MLM trajectories.

**H_2 (regime-invariance of shape).** Slopes differ across regimes by ≥ 2×. Residual structure is statistically indistinguishable across regimes (ANOVA p > 0.05).

**H_3 (alternative: shape differs across regimes).** If H_2 fails — residual structure differs across regimes — fit a regime classifier on shape features. Target: ≥ 90% cross-validated accuracy.

**H_2 and H_3 are alternatives.** Either is informative:
- H_2 holds → exponential baseline validated → Experiment 5 (anomaly detection) is on solid theoretical footing.
- H_3 holds → shape-of-decay is itself a label-free regime classifier → second independent solution to §3.3.

**Falsification criterion:** If both H_2 and H_3 fail (residuals differ across regimes but classifier accuracy < 70%), the temporal-shape approach is inconclusive.

## Predicted outcome (from existing §4.3.1 + §4.6 data)

From Fardini Doc 1: NPS velocity profile for §4.3.1 MLM Wikipedia is approximately {−0.117, −0.145, −0.049, −0.003, +0.0002, −0.0001}. Peak at epochs 1→3, then exponential-like decay.

For §4.6 anti-InfoNCE: NPS goes 1.0 → 0.118 over 50 epochs — much larger overall movement, likely different decay shape (controlled overshoot vs natural plateau).

For §4.1 contrastive LoRA on E5: NPS endpoint 0.60 (small relative drift), trajectory likely much flatter.

Strong prior that **shape differs across regimes** — best estimate H_3 wins, giving a second §3.3 attack independent of Experiment 1.

## Compute

< 1 CPU-hour. Pure post-hoc analysis of existing checkpoint trajectories.

## Paper integration

- **New §3.6 "Temporal-shape regime classification"** (if H_3 wins) or **§4.3 augment** (if H_2 wins).
- **§7.4 partial resolution** — temporal shape replaces the broken NPS-bound framing with a regime-classifier framing.
- Pairs with Experiment 1 — if both succeed, the paper has two independent label-free regime classifiers from different signal axes.

---

# Experiment 3 — Trustworthiness/Continuity asymmetry

**Source:** my Wave 1, repositioned. **Cheap test of §3.3 attack.**

NPS lumps two failure modes together: false neighbors introduced and true neighbors lost. The ratio of these — Trustworthiness / Continuity — may discriminate regimes that NPS magnitude cannot.

## Hypothesis

At matched NPS, the T/C ratio differs across regimes. Specifically: anti-aligned drift introduces more false neighbors (T low, C high → ratio < 1); aligned drift loses more true neighbors that were intentionally restructured (T high, C low → ratio > 1).

## Setup

Same checkpoint inventory as Experiment 1.

## Procedure

1. Compute Trustworthiness and Continuity at each checkpoint. Lib deps: **B1, B2**.
2. Compute T/C ratio.
3. Compare ratio means ± std across regimes.

## Pre-registration

**H_TC (asymmetry separates regimes).** At epoch 50:
- Aligned regime: T/C > 1.1 (in any direction)
- Anti-aligned regime: T/C < 0.9
- Separation ≥ 2σ at the multi-seed level

**Falsification:** If T/C ∈ [0.9, 1.1] across all regimes, NPS's directional decomposition is symmetric and doesn't carry regime information.

## Predicted outcome

Genuinely uncertain. The decomposition could either reveal a clean asymmetry or show that NPS already captures the relevant local information. Worth running because the cost is trivial.

## Compute

< 1 GPU-hour given existing k-NN indexes.

## Paper integration

- **§4.3 augment** — add T/C columns to Tables 8, 9.
- If positive: new lightweight regime indicator (cheaper than Experiment 1's behavioral signal).
- If negative: a falsifying test that informs the orthogonality argument in Experiment 1's §6.2.

---

# Experiment 4 — Plateau-detection validation across regimes

**Source:** Fardini Doc 1 §4.1. **MLOps contribution.**

Validate the 97%-at-20%-compute claim across all three regimes, not just MLM Wikipedia. If plateau detection is regime-robust, it's a deployable training-loop feature; if it's regime-specific, that's also informative.

## Hypothesis

A plateau criterion `|velocity| < ε AND |acceleration| < δ for k consecutive checkpoints` (default ε=0.005, δ=0.001, k=3) fires before epoch 15 across all multi-checkpoint regimes, capturing ≥ 95% of eventual geometric change in each.

## Setup

Same multi-checkpoint trajectories as Experiment 2.

## Procedure

1. For each trajectory, evaluate the plateau criterion at every checkpoint. Lib dep: **F3**.
2. Record `plateau_epoch` (first checkpoint where criterion fires for k=3 consecutive).
3. Record `geometric_change_captured = (NPS[plateau_epoch] - NPS[0]) / (NPS[end] - NPS[0])`.
4. Per regime: mean ± std of (plateau_epoch, captured fraction) across seeds and families.

## Pre-registration

**H_plateau (regime-robust).** For each regime, plateau_epoch ≤ 15 with σ ≤ 5 across seeds, and captured fraction ≥ 95% with σ ≤ 3 pp.

**H_plateau_aligned_only.** Plateau detection works on aligned regime but fails on anti-aligned (which keeps moving). Partial result.

**Falsification:** captured fraction < 80% in any regime, or plateau_epoch σ > 10 across seeds.

## Predicted outcome (from §4.3.1)

For MLM Wikipedia: plateau detected at epoch 10, captured fraction ≈ 97% — confirmed in the Fardini Doc 1 worked example.

For aligned regime: NPS moves more slowly (endpoint 0.60 vs MLM's 0.48), but trajectory should still plateau. Probably plateau_epoch ∈ [10, 20].

For anti-InfoNCE: trajectory drives NPS to 0.118 — likely *no* plateau by epoch 50 (the regime has no intrinsic ceiling per §4.6.1). This would partially refute H_plateau but is itself a finding: **the no-ceiling property of anti-aligned drift means plateau detection works as a regime indicator** — failing to plateau is a signal.

## Compute

< 1 CPU-hour.

## Paper integration

- **§4 sidebar or footnote.** "Plateau detection generalizes across regimes, except in the anti-aligned no-ceiling case (§4.6.1), where failure-to-plateau is itself a regime signal."
- **§7.5 infrastructure update.** Plateau detection becomes a v0.2 library feature.

---

# Experiment 5 — Anomaly detection with anti-InfoNCE positive control

**Source:** Fardini Doc 1 §4.2 + the §4.6 anti-InfoNCE checkpoints.

The missing piece in Fardini Doc 1's anomaly-detection proposal was a positive control — a known-bad training trajectory to validate the criterion against. The §4.6 anti-InfoNCE run is that control: deliberately maximally-anti-aligned, drives NPS to 0.118 (overshoot beyond the pre-registered [0.40, 0.55] window).

## Hypothesis

Fitting log|velocity| to a line in the post-warmup phase of the §4.1 aligned trajectories gives a residual standard deviation σ_normal. The §4.6 anti-InfoNCE trajectory produces residuals exceeding 3σ_normal at one or more post-warmup checkpoints, correctly flagging it as anomalous.

## Setup

- **Reference distribution (healthy)**: §4.1 contrastive LoRA on E5, CLIP, BERT — multi-family, 9 seeds total.
- **Test distribution (suspected anomalous)**: §4.6 anti-InfoNCE on E5 — 3 seeds.

## Procedure

1. For each reference trajectory, fit log|velocity| of NPS to a line for epochs ≥ 3. Lib dep: **F4**.
2. Compute residual std σ_normal pooled across reference trajectories.
3. For each anti-InfoNCE trajectory, compute log|velocity| residuals against the same reference fit (or against its own line).
4. Flag any checkpoint with residual > 3σ_normal.

## Pre-registration

**H_anomaly_works.** All 3 anti-InfoNCE seeds produce ≥ 1 checkpoint flagged at 3σ. False-positive rate on the 9 reference seeds: 0 (no aligned trajectory is flagged).

**H_anomaly_partial.** True-positive rate ≥ 2/3 seeds flagged; false-positive rate ≤ 1/9 seeds.

**H_anomaly_fails.** True-positive < 2/3 or false-positive > 2/9.

**Falsification:** the criterion either fires on aligned (false positive) or misses anti-InfoNCE entirely.

## Predicted outcome

The anti-InfoNCE trajectory is qualitatively different from MLM and aligned — it drives NPS to 0.118 over 50 epochs with monotonically decreasing adversarial loss (§4.6). Strong prior that velocity shape is non-exponential. **High confidence H_anomaly_works**.

## Compute

< 1 CPU-hour.

## Paper integration

- **§4.6 augment.** "Anomalous training dynamics can be detected from velocity-profile residuals alone, without any reference to the loss function itself."
- **§7.5 library item.** Anomaly detection ships in v0.3 with the multi-seed reference distribution as default calibration.

---

# Experiment 6 — Per-regime transfer functions (§7.4 NPS-bound replacement)

**Source:** §7.4 open question + my I1.

The §7.4 open question: *"What replaces the conjectured NPS bound? An earlier version of this work conjectured a lower bound degradation ≥ (1 − NPS); the bound is strictly violated under aligned drift and meaningless under anti-aligned drift."*

The replacement is **regime-aware empirical transfer functions** — one calibrated LinearTransfer per regime, with regime determined by Experiment 1 or 2's classifier.

## Hypothesis

Fitting a separate LinearTransfer per regime (inputs: NPS, CKA, isotropy_delta, behavioral signal from Exp 1; output: labeled benchmark drop) yields:
- Per-regime R² ≥ 0.85
- Cross-regime R² (pooled fit, no regime distinction) ≤ 0.5
- Regime-aware predictions within ± 2 pp of measured benchmark drops on held-out seeds

## Setup

Use existing §4 checkpoints with labeled benchmark scores. Each (checkpoint, regime, family, seed) tuple is a data point.

Training: 2 seeds per (regime, family). Held-out: 1 seed per (regime, family).

## Procedure

1. For each regime, fit `LinearTransfer` on (drift metrics, behavioral signal) → (labeled benchmark drop). Lib dep: **I1**.
2. Compare per-regime R² to pooled (regime-blind) R².
3. Evaluate predictions on held-out seeds.
4. Per-regime feature importance — which signal dominates the prediction in each regime?

## Pre-registration

**H_per_regime_works.** Per-regime R² ≥ 0.85; pooled R² ≤ 0.5; held-out prediction error ≤ 2 pp on average.

**H_pooled_works.** Pooled R² ≥ 0.85 — meaning the regime distinction adds nothing beyond what the joint feature vector already carries.

**H_neither.** Both R² < 0.85 — regression framing is inadequate; need a different transfer family.

**Falsification:** If H_per_regime_works holds but per-regime feature importance is essentially identical across regimes, the regime-separation is cosmetic and a single transfer with regime-as-feature is preferred.

## Predicted outcome

Strong prior on H_per_regime_works given the §3 evidence that the regimes are categorically distinct. The interesting question is per-regime feature importance — does the anti-aligned regime weight behavioral signal more than NPS, while the aligned regime weights NPS more? That would be the cleanest mechanistic story.

## Compute

< 1 GPU-hour given existing checkpoints + benchmark scores.

## Paper integration

- **§2 calibration update.** Severity scale becomes regime-conditional (per-regime thresholds + per-regime transfer function).
- **§7.4 resolution.** "The NPS-bound framing is replaced by regime-aware empirical transfer functions calibrated against multi-seed reference runs."
- **Headline claim refinement.** "The framework predicts deployment-relevant drift outcomes from purely label-free signals, with regime classification handled by joint-trajectory analysis and per-regime transfer functions handling magnitude."

---

# Experiment 7 — Anchor-set sensitivity sweep

**Source:** §5.2 extension.

§5.2 already establishes 2.3× NPS magnitude variation between training-distribution and OOD anchors. This experiment extends the finding to all proposed metrics and the behavioral signal, providing a guidance table for production use.

## Hypothesis

The anchor-set distribution effect generalizes across metrics: each metric has a characteristic sensitivity coefficient (ratio of magnitude under training-dist vs OOD anchor) that's stable across regimes and families.

## Setup

Use existing §4.1 CLIP and §4.3.1 E5 checkpoints (which have both training-dist and OOD anchor evaluations per §5.2).

Add: a third anchor distribution per family — random web-scraped text/images that match neither training nor evaluation.

## Procedure

1. Compute every Family A/B/C/D/E metric (those that survive Family-A's Q/D split requirement) at each checkpoint × each anchor.
2. Per metric, compute the sensitivity coefficient: max magnitude / min magnitude across anchors.
3. Tabulate.

## Pre-registration

**H_sensitivity_table.** Each metric has a sensitivity coefficient within ± 30% across the two regimes (aligned, anti-aligned) and two families (E5, CLIP).

## Compute

~2 GPU-hours (need to encode 3 anchors × 2 families × many checkpoints).

## Paper integration

- **§5.2 augment.** Replace single-metric finding with a table: per-metric anchor-distribution sensitivity coefficient, with recommended anchor protocols.
- **§5.4 (new) Metric robustness guidance.** Production users get a clear table mapping (metric, anchor-distribution) to expected magnitude band.

---

# Experiment 8 — Probe-methodology second-task replication

**Source:** §5.1 extension.

§5.1 reports a 28.5 pp Banking77 probe-config swing (63.5% weak → 92.05% strong). This is a major methodology contribution but rests on a single dataset. Replicate on one additional classification benchmark to strengthen it.

## Hypothesis

The same probe-configuration sensitivity holds on a second standard classification benchmark: weak probe (2K samples, fixed C=1.0) underperforms strong probe (full training set, LogisticRegressionCV) by ≥ 15 pp at identical embeddings.

## Setup

- Second benchmark: AG News (4-class text classification, larger training set) and/or Banking77 itself with a different embedding (BGE instead of E5) as a robustness check.
- Same probe configurations as §5.1.

## Procedure

1. Encode the second benchmark's training set with E5-base-v2 (and BGE separately).
2. Fit weak probe (2K samples, C=1.0) and strong probe (full set, LogisticRegressionCV).
3. Report accuracy delta.

## Pre-registration

**H_replicates.** Delta ≥ 15 pp on the second benchmark with E5; delta ≥ 15 pp on Banking77 with BGE.

**Falsification:** Delta < 10 pp — would suggest Banking77 + E5 was an unusual pairing.

## Compute

~2 GPU-hours.

## Paper integration

- **§5.1 augment.** Single-task finding becomes cross-task replication. Strengthens the methodology contribution from "documented on one benchmark" to "reproducible across benchmarks and embeddings."

---

# Falsification map

A guide to what each negative outcome means for the paper.

| Experiment | If positive | If negative |
|---|---|---|
| 1 (behavioral) | New §3.5; resolves §3.3 | Report H_C; orthogonality limit confirmed; pursue A7/A8 in future work |
| 2 (decay shape) | New §3.6; second §3.3 resolution | Confirms exponential baseline (validates Exp 5) — still useful |
| 3 (T/C) | Cheap lightweight regime indicator | Eliminates a hypothesis cleanly |
| 4 (plateau) | MLOps contribution; library feature | Anti-aligned no-plateau finding is itself a result |
| 5 (anomaly) | Anomaly-detection library feature | Need different anomaly criterion; exponential baseline insufficient |
| 6 (transfer fn) | Resolves §7.4 cleanly | Different transfer family needed; report what failed |
| 7 (anchor sweep) | Methodology contribution | Anchor effect is metric-specific; needs case-by-case treatment |
| 8 (probe replication) | Strengthens §5.1 | Banking77+E5 was an unusual pairing; weaken claim |

**No experiment has a pure "wasted compute" outcome.** All are pre-registered with informative negative results.

---

# Compute budget summary

| Experiment | Compute |
|---|---|
| 1 — Joint trajectory | ~3 GPU-hrs |
| 2 — Decay shape | <1 CPU-hr |
| 3 — T/C asymmetry | <1 GPU-hr |
| 4 — Plateau validation | <1 CPU-hr |
| 5 — Anomaly detection | <1 CPU-hr |
| 6 — Per-regime transfer | <1 GPU-hr |
| 7 — Anchor sweep | ~2 GPU-hrs |
| 8 — Probe replication | ~2 GPU-hrs |
| **Total** | **~9 GPU-hrs + few CPU-hrs** |

Comparable to a single LoRA fine-tune of the size used in §4. Substantially less than any of the existing experiments.

---

# Submission packaging

## Workshop draft (immediate)

Include Experiments 1, 2, 3, 4 as the main contributions beyond the current draft. The §3.3 resolution from Experiment 1 alone is workshop-worthy. Experiments 2–4 provide breadth.

## Main-track strengthening

Add Experiments 5, 6, 7, 8. Experiment 6 (NPS-bound replacement) resolves the §7.4 open question concretely. Experiment 7 strengthens the §5.2 methodology contribution to a general guidance table.

## Recommended priority order

If running them sequentially:

1. **Experiment 1 first.** Highest leverage; binary decisive outcome; closes §3.3 if H_A.
2. **Experiment 2 in parallel** (no GPU needed). Either validates Exp 5's theoretical baseline or provides an independent §3.3 attack.
3. **Experiment 3** (cheap). Either provides a lightweight regime indicator or cleanly eliminates a hypothesis.
4. **Experiments 4 + 5** together (use Exp 2's velocity profiles as the input). MLOps + mechanism contributions.
5. **Experiment 6.** Resolves §7.4 once Exp 1 has run (needs the behavioral signal as an input).
6. **Experiment 7** if time permits before main-track submission.
7. **Experiment 8** if time permits.

## Pre-registration practice

Following the project's existing pre-registration practice (§7.2, §4.3.2, §4.4): for each experiment, write the pre-registration document **before running** with the hypotheses, falsification criteria, and predicted outcomes specified above. Commit to `experiments/<exp_name>/PREREGISTRATION.md` before the first run. Match the §4.6 anti-InfoNCE template.
