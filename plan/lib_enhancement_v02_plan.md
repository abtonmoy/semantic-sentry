# lib_enhancement v0.2 — implementation plan

Companion to `lib_enhancement.md`. This doc is the *execution* schedule
for the 17 items in the v0.2 wave (3 infra + 5 behavioral + 3 NPS-family
+ 4 temporal + 1 calibration + 1 multi-k registration helper).

Scope decisions:
- Stay strictly inside the v0.2 wave declared in `lib_enhancement.md`.
- Don't ship breaking API changes; everything additive on top of the
  cleaned v0.1.x surface (`MetricRegistry`, `AnchorSet`, `DriftMonitor`,
  `Comparison`).
- One PR per wave below — keeps reviewable diffs ~300–600 LOC.
- Each item ships with: implementation + 1–3 unit tests + a one-line
  CHANGELOG entry. Aggregate to ~127 → ~170 unit tests after v0.2.

---

## Dependency graph (must be respected)

```
G1 (params dict)  ──┬──►  A1 A2 A3 A4 A5     (Family A, behavioral)
                    └──►  B1 B2 B5            (Family B, priority NPS)
G2 (multi-k helper) ─────►  B5  + future A5_at_k
H1 (Q/D split)      ─────►  A1 A2 A3 A4
H2 (anchor tag)     ─────►  consumed by I2
G4 (temporal wrapper) ───►  F1 F2 F3
F1 (velocity)       ─────►  F2 (acceleration)
F1 + F2             ─────►  F3 (plateau detector)
I2 (noise-floor severity) — independent, can land any time
```

Read top-to-bottom in the wave list below: each wave only depends on
items shipped in earlier waves.

---

## Wave 1 — Infrastructure (must ship first)

Order matters; each unblocks downstream families.

### G1 — Hyperparameter dict on `MetricEntry`
- **Files:** `metrics/registry.py`
- **Change:** Add `params: dict[str, Any] = field(default_factory=dict)` to
  `MetricEntry`. Update `MetricRegistry.register(name, fn, *, params={},
  range=None, description=None)`. In `_call_metric()` (the call path used
  by `compute_all`), do `fn(Z0, Z1, **entry.params)`.
- **Back-compat:** Existing built-ins call with no params; `params={}`
  default keeps them untouched.
- **Tests:** Round-trip a metric registered with `params={"k": 5}` and
  confirm `compute_all` invokes it with `k=5`.
- **Effort:** ~50 LOC + 1 test.

### G2 — Multi-k registration helper
- **Files:** `metrics/registry.py`
- **Change:** Add `register_at_k(base_name, fn, ks=(1, 5, 10, 25, 50, 100),
  range=None, description=None)` that calls `register(f"{base_name}_at_{k}",
  fn, params={"k": k}, range=range, description=...)` for each k.
- **Depends on:** G1 (uses `params={"k": k}`).
- **Tests:** `register_at_k("nps", nps, ks=(5, 10))` produces `nps_at_5`
  and `nps_at_10` entries that both work.
- **Effort:** ~30 LOC + 1 test.

### H1 — Q/D partition on `AnchorSet`
- **Files:** `probes/anchor_set.py`
- **Change:** Add
  ```python
  def partition(self, ratio: float = 0.5, seed: int = 0) -> tuple["AnchorSet", "AnchorSet"]:
      ...
  ```
  Deterministic shuffle with `seed`; first `int(n * ratio)` indices →
  query subset, rest → document subset. Both inherit from the parent's
  `version_hash` plus a `(seed, ratio)` suffix so paired snapshots can
  detect mismatched splits.
- **Storage:** Add a `parent_version_hash: str | None` field that the new
  partitions populate; existing `AnchorSet`s leave it as `None`.
- **Tests:** Same `(ratio, seed)` always returns the same split; cross-
  check that `Q ⊕ D == original` (as multisets); `parent_version_hash`
  matches across the two children.
- **Effort:** ~60 LOC + 2 tests.

### H2 — Anchor distribution provenance tag
- **Files:** `probes/anchor_set.py`, `core/snapshot.py`, `core/comparison.py`
- **Change:** New optional field `distribution_tag: str | None = None`
  on `AnchorSet`. Snapshot copies it into its metadata (`metadata["anchor_distribution_tag"]`).
  `Comparison` exposes it via `comparison.metadata["anchor_distribution_tag"]`
  on compare-time pass-through.
- **Tests:** Round-trip the tag through `Snapshot.save/load` + `compare()`.
- **Effort:** ~40 LOC + 1 test.

### I2 — Multi-seed noise-floor severity calibration
- **Files:** `core/comparison.py` (existing severity property), new module
  `core/severity_calibration.py`.
- **Change:**
  ```python
  def calibrate_thresholds(reference_comparisons: list[Comparison],
                          *, low_q: float = 0.84, medium_q: float = 0.50,
                          high_q: float = 0.16) -> dict[str, float]:
      """Derive nps_low/medium/high + cka_low/medium/high from quantiles
      of the metric values in `reference_comparisons` (typically
      same-seed-vs-same-seed runs that should be 'no drift')."""
  ```
  Returns the same threshold dict shape `Comparison.thresholds` already
  consumes. Backward-compatible: callers who don't use this still get the
  hard-coded defaults.
- **Tests:** Synthesize 10 "no-drift" comparisons with NPS ~ 1 - N(0, σ),
  call `calibrate_thresholds`, verify thresholds widen with σ.
- **Effort:** ~80 LOC + 2 tests.

### G4 — Temporal wrapper for any base metric
- **Files:** new module `metrics/temporal.py`
- **Change:**
  ```python
  class TemporalSeries:
      """Wraps a registered metric as a time-series operator."""
      def __init__(self, metric_name: str): ...
      def values(self, snapshots: list[Snapshot], times: list[float]) -> np.ndarray:
          """Return per-checkpoint metric values: M(s_0, s_t) for each t."""
      def velocity(self, snapshots, times) -> np.ndarray: ...   # Family F1
      def acceleration(self, snapshots, times) -> np.ndarray: ... # F2
      def plateau_mask(self, snapshots, times, *,
                       eps=0.005, delta=0.001, k=3) -> np.ndarray: ... # F3
  ```
  Lives in `metrics/` because it's metric-aware (looks up `MetricEntry`
  from the global registry).
- **Tests:** Synthetic monotone-decreasing trajectory → velocity is
  negative; flat trajectory → plateau_mask returns all True after the
  first window.
- **Effort:** ~120 LOC + 3 tests.

**Wave 1 total:** ~380 LOC + 10 tests. ~1.5 days.

---

## Wave 2 — Behavioral / ranking-stability (Family A)

Depends on G1, G2 (params), H1 (Q/D split). Order: simplest → most
involved so each PR is independently mergeable.

### A5 — Self-retrieval top-k consistency
- **Files:** new `metrics/behavioral.py`, register in `metrics/registry.py`
- **Function signature:** `top_k_set_consistency(Z0, Z1, k: int = 10) -> float`.
  Returns fraction of rows whose top-k set in Z0 equals top-k set in Z1.
- **Reuses:** `_get_knn_indices` from `metrics/nps.py` (already self-
  exclusion-correct after the v0.1.x fix).
- **Multi-k registration:** `register_at_k("top_k_consistency",
  top_k_set_consistency, ks=(1, 5, 10))` via G2.
- **Tests:** `top_k_set_consistency(Z, Z, k=10) == 1.0`; on a 50-row
  random matrix vs a permuted copy, score is < 0.05.
- **Effort:** ~40 LOC + 2 tests.

### A1 — Score-distribution JSD
- **Files:** `metrics/behavioral.py`
- **Function signature:** `score_jsd(Z0, Z1, *, q_idx: np.ndarray | None,
  d_idx: np.ndarray | None, n_bins: int = 100) -> float`.
- **Implementation:** Cosine sim matrix (`Z[q_idx] @ Z[d_idx].T`) per
  snapshot; histogram both into `n_bins` bins on the shared `[-1, 1]`
  range; compute JSD with `scipy.spatial.distance.jensenshannon`.
- **Q/D wiring:** Caller supplies `q_idx`/`d_idx` via `MetricEntry.params`
  (G1). `DriftMonitor.compare_with_partition()` (a thin new helper)
  builds these from `H1.partition()` indices.
- **Tests:** JSD == 0 when Z0 == Z1; JSD > 0.1 on two random matrices
  with different scales.
- **Effort:** ~80 LOC + 2 tests.

### A2 — Mean abs pointwise score Δ
- **Files:** `metrics/behavioral.py`
- **Function signature:** `mean_abs_score_delta(Z0, Z1, *, q_idx, d_idx,
  normalise: bool = True) -> float`.
- **Reuses:** Same similarity-matrix code as A1; pull into a shared
  `_pairwise_cosine(Z, q_idx, d_idx)` helper at module scope.
- **Tests:** Identical → 0; norm-only-changed Z1 (Z1 = 2 * Z0 normalised
  to unit) → 0 because cosine sim is scale-invariant.
- **Effort:** ~30 LOC + 1 test.

### A3 — Per-query Rank-Biased Overlap (RBO)
- **Files:** `metrics/behavioral.py`
- **Function signature:** `mean_rbo(Z0, Z1, *, q_idx, d_idx, p: float = 0.9,
  k: int | None = None) -> float`.
- **Implementation:** Vendor a small RBO implementation (~30 LOC) — no
  good single-purpose dep. Use the standard formula
  `RBO_p = (1 - p) * Σ_{d=1..D} p^(d-1) * (|A_d ∩ B_d| / d)` over the
  truncation depth `D = k or |D|`.
- **Tests:** Identical rankings → RBO = 1; reversed rankings → RBO
  approaches `1 − p` (well below 0.5 for default `p=0.9`).
- **Effort:** ~80 LOC + 2 tests.

### A4 — Per-query Kendall τ
- **Files:** `metrics/behavioral.py`
- **Function signature:** `mean_kendall_tau(Z0, Z1, *, q_idx, d_idx) -> float`.
- **Implementation:** `scipy.stats.kendalltau` per query, return mean.
  No vendored algorithm needed; `scipy` is already a dep.
- **Tests:** Identical rankings → τ = 1; reversed rankings → τ = -1.
- **Effort:** ~25 LOC + 2 tests.

**Family A registration:** Each of A1–A5 registers with `params` carrying
default Q/D indices (full anchor → 50/50 split with seed=0 if the
`AnchorSet` has not been pre-partitioned).

**Wave 2 total:** ~255 LOC + 9 tests. ~1.5 days.

---

## Wave 3 — NPS-family priority (B1, B2, B5)

Depends on G1, G2.

### B1 — Trustworthiness
- **Files:** new `metrics/nps_family.py`, register in registry
- **Function signature:** `trustworthiness(Z0, Z1, k: int = 10) -> float`.
- **Implementation:** For each i, find top-k in Z1; for each "false
  neighbour" j in Z1's top-k that's not in Z0's top-k, accumulate the
  rank penalty (rank_in_Z0(j) - k) when positive. Normalise.
- **Reuses:** `_get_knn_indices` from `nps.py`.
- **Tests:** `trustworthiness(Z, Z) == 1.0`; on synthetic with introduced
  hub points, score drops.
- **Effort:** ~70 LOC + 2 tests.

### B2 — Continuity
- **Files:** `metrics/nps_family.py`
- **Function signature:** `continuity(Z0, Z1, k: int = 10) -> float`.
- **Implementation:** Mirror of B1 with Z0/Z1 swapped — penalise true
  neighbours from Z0 that fall out of Z1's top-k.
- **Tests:** Symmetry check `continuity(Z0, Z1) == trustworthiness(Z1, Z0)`.
- **Effort:** ~40 LOC + 1 test (mostly factored from B1).

### B5 — NPS curve (multi-k)
- **Files:** `metrics/registry.py` boot path
- **Change:** Inside `_register_builtins()`, call
  `register_at_k("nps", nps, ks=(1, 5, 10, 25, 50, 100))` so each k is
  a distinct registry entry. The existing `nps` registration stays for
  backward-compat (defaults to k=10).
- **Tests:** All six entries are present after construction; each calls
  through to `nps` with the right `k`.
- **Effort:** ~20 LOC (wiring only) + 1 test.

**Wave 3 total:** ~130 LOC + 4 tests. ~1 day.

---

## Wave 4 — Temporal layer (F1, F2, F3)

Depends on G4. F1 → F2 → F3 in order; all live in the `TemporalSeries`
class shipped by G4.

### F1 — Velocity wrapper
- **Files:** `metrics/temporal.py` (`TemporalSeries.velocity`)
- **Implementation:** Central differences, `(M[t+1] - M[t-1]) / (t[t+1]
  - t[t-1])`. Forward/backward diff at endpoints. Handle unequal
  spacing.
- **Tests:** Linear trajectory `M[t] = a + b*t` → velocity is `b`
  everywhere (within numerical tolerance).
- **Effort:** ~30 LOC + 1 test.

### F2 — Acceleration wrapper
- **Files:** `metrics/temporal.py` (`TemporalSeries.acceleration`)
- **Implementation:** Apply `velocity()` twice. ~5 LOC.
- **Tests:** Quadratic trajectory → constant acceleration.
- **Effort:** ~10 LOC + 1 test.

### F3 — Plateau detector (priority)
- **Files:** `metrics/temporal.py` (`TemporalSeries.plateau_mask`)
- **Implementation:**
  ```python
  vel = self.velocity(snapshots, times)
  acc = self.acceleration(snapshots, times)
  flat = (np.abs(vel) < eps) & (np.abs(acc) < delta)
  # require k consecutive True values
  return _k_run_mask(flat, k)
  ```
  Default thresholds per `lib_enhancement.md` (ε=0.005, δ=0.001, k=3).
- **Tests:** Flat-after-warmup synthetic trajectory → mask is False for
  warmup checkpoints, True afterward; oscillating trajectory → mask
  stays False.
- **Effort:** ~40 LOC + 2 tests.

**Wave 4 total:** ~80 LOC + 4 tests. ~0.5 day.

---

## Cross-cutting deliverables

These ride alongside whichever wave touches them.

- **`compute_all` extension** to honour `params` (Wave 1 / G1).
- **`compare()` Q/D wiring** — `DriftMonitor.compare(s0, s1, *,
  partition: tuple[AnchorSet, AnchorSet] | None = None)`. When
  `partition` is supplied, the registry sees Q/D indices in
  `MetricEntry.params` and Family A metrics activate. (Wave 2.)
- **README + CHANGELOG entries** per wave.
- **`plan/lib_enhancement.md` checkbox tick-offs** per item shipped.

---

## What's deferred from this plan

Per `lib_enhancement.md`'s phasing:

- A6, B3, B4, B6, B7, C1–C3, D1–D3 D5, E4, F4, F5, G3, G5, I1, I3 → v0.3.
- A7, A8, C4–C6, D4, D6, E1–E3, E5, H3, H4 → future / research.

Don't pull these forward unless a concrete v0.2 user surfaces.

---

## Effort summary

| Wave | LOC | Tests | Days |
|---|---|---|---|
| 1 — Infra (G1, G2, H1, H2, I2, G4) | ~380 | 10 | 1.5 |
| 2 — Behavioral (A1–A5) | ~255 | 9 | 1.5 |
| 3 — NPS family (B1, B2, B5) | ~130 | 4 | 1.0 |
| 4 — Temporal (F1, F2, F3) | ~80 | 4 | 0.5 |
| Cross-cutting wiring | ~80 | 2 | 0.5 |
| **v0.2 total** | **~925** | **29** | **~5 days** |

Test count target: 127 (current) → **~156**.

---

## Open decisions to confirm before starting

1. **Q/D split default ratio** — `lib_enhancement.md` says 0.5; lock it
   here. Configurable via `partition(ratio=...)`.
2. **RBO `p`** — default 0.9 per the doc; confirm.
3. **Plateau-detector defaults** — ε=0.005, δ=0.001, k=3 per the doc;
   tunable per metric since "small velocity" depends on the metric scale.
4. **Behavioral metric naming convention** —
   `score_jsd` / `mean_abs_score_delta` / `mean_rbo` / `mean_kendall_tau`
   / `top_k_consistency_at_{k}`. Keep the verb form so the registry
   reads as `compute_all` outputs.
5. **Where does `TemporalSeries` live?** Recommend `metrics/temporal.py`
   (accessed as `from semantic_sentry.metrics.temporal import
   TemporalSeries`). Alternative: `core/temporal.py` if it grows
   non-metric features.

If any of these need to change, edit before Wave 1 starts.

---

## Verification (per wave)

- Wave 1: `pytest tests/unit/ -q` → 127 + 10 = 137 pass.
- Wave 2: 137 + 9 = 146 pass.
- Wave 3: 146 + 4 = 150 pass.
- Wave 4: 150 + 4 = 154 pass; `mypy src/semantic_sentry` continues to
  pass (no new strict-mode regressions).
- After all waves: `from semantic_sentry import DriftMonitor` import
  smoke; `from semantic_sentry.metrics.temporal import TemporalSeries`
  import smoke; `MetricRegistry().list_metrics()` shows the new entries.
- Whole-paper experiments under `../experiments/` continue to run without
  modification (additive-only API).
