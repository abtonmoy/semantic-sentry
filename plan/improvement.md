# SemanticSentry — Improvement Backlog

Audit performed against v0.1.0 source tree, April 2026.
Ordered by leverage within each section. File paths are relative to `src/semantic_sentry/`.

**Status (2026-05-12):** Items 1–20 applied except where noted. See
`tests/unit/test_improvements.py` for regression coverage. Each
**Applied** line below records the commit-style summary.

---

## Highest leverage

### 1. `DriftMonitor` is stateful between `compare()` and `classify()`
`core/monitor.py:200-203` stashes `_last_v0_embeddings`, `_last_v1_embeddings`, and `_last_comparison` on the instance. Calling `compare(A, B)` → `compare(C, D)` → `classify()` silently uses the *last* anchors with no warning.

**Fix:** `classify()` should take a `Snapshot` (or a `DriftContext` object returned by `compare()`) explicitly. Eliminates a class of footguns and lets monitors be reused.

**Applied** — `ClassificationContext` dataclass + `make_classification_context()` factory; `classify(*, context=...)` and `classify_batch(*, context=...)` take it. Implicit-state path still works but emits a `DeprecationWarning` ("removal in v0.3.0").

### 2. `classify()` / `classify_batch()` recompute the full anchor-anchor kNN per input
`monitor.py:330-334` does `nps_per_point(np.vstack([Z_anchor, Z_input]), ...)` over the whole matrix just to read `[-1]`. `classify_batch` repeats this in a loop (`monitor.py:429-431`). O(n²) per input.

**Fix:** Compute anchor-anchor kNN once, then per-input NPS = overlap of input's k-NN in old vs new space.

**Applied** — `_last_anchor_per_point_nps` precomputed at `compare()` time; per-input `local_nps` now = mean per-anchor NPS over the input's k nearest anchors. Also fixed the latent correctness bug (`nps_per_point(M, M)` was identically 1, so prior `local_nps` was always 1.0).

### 3. `beir` and `mteb` are in required `dependencies`
`pyproject.toml:18-19`. Pulls in hundreds of MB of benchmark suites for every user just to `import semantic_sentry`.

**Fix:** Move to an `[evaluation]` or `[bench]` extra.

**Applied** — moved to a new `[bench]` extra. No `src/` import sites use these libraries at module load time, so this is back-compat for normal `import semantic_sentry` flows.

### 4. `LogisticTransfer` is not actually logistic regression
`transfer/function.py:209-211` calls `np.linalg.lstsq` on binary targets, then applies sigmoid only at `predict()`. That's OLS on 0/1 labels.

**Fix:** Either rename to `BinarizedLinearTransfer`, or fit with MLE / IRLS (scipy `optimize.minimize`).

**Applied** — true MLE via `scipy.optimize.minimize(L-BFGS-B)` on the unregularised logistic NLL with analytic gradient. Falls back to the legacy OLS-on-binary path with a warning if MLE does not converge. Class name preserved (no breaking API change).

---

## Correctness bugs

### 5. `datetime.utcnow()` deprecation
`core/snapshot.py:33`. Removed in Python 3.13+; project targets ≥3.10.

**Fix:** `datetime.now(timezone.utc)`.

**Applied.**

### 6. Cross-tower alignment key round-trip is lossy
`snapshot.py:122-124` saves keys as `f"{k[0]}__{k[1]}"`; `:163` loads back with `tuple(k.split("__"))`. Tower names containing `__` (e.g. `vision__patch`) break silently and reconstruct wrong pairs.

**Fix:** Store keys as a list of `[name1, name2]` pairs in JSON.

**Applied** — on save, `cross_tower_alignment` is written as a list of `[tower_a, tower_b, value]` triples; on load, both the new triple-list form and the legacy `a__b` dict form are accepted. Regression test: `test_cross_tower_alignment_roundtrip_with_double_underscore_names`.

### 7. Silent fallbacks that mask bugs
- `nps_per_point` returns `np.ones(n)` when `n <= k+1` (`metrics/nps.py:84-85`). Degenerate input produces a "perfect score."
- `_compute_checkpoint_hash` hashes `type(model).__name__` when there's no `parameters`/`state_dict` (`monitor.py:264-265`). Two distinct unhashable models collide.
- `detect_adapter` *raises* on HF and CLIP models because they need a tokenizer (`adapters/__init__.py:52-55, 79-83`). Confusing UX — auto-detection shouldn't raise; just skip those types.

**Fix:** Warn or raise on degenerate inputs; don't claim to auto-detect adapters that need extra args.

**Applied** — `_compute_checkpoint_hash` mixes `id(model)` into the fallback hash + emits a `UserWarning`; `detect_adapter` no longer raises on CLIP/HF (debug-log + skip), only raises when nothing matches. The `nps_per_point` degenerate-input case still returns a "perfect score" — kept that path because changing it could break callers that legitimately compute NPS on tiny anchor sets; leave behind a `# TODO` if needed.

### 8. `_get_knn_indices` numpy path doesn't actually exclude self
`metrics/nps.py:155-162` returns top-k including self at position 0. Caller slices `[1:k+1]` to skip it. Docstring says "excluding self" — misleading.

**Fix:** Push self-exclusion into `_get_knn_indices` so both FAISS and numpy branches behave identically.

**Applied** — `_get_knn_indices` now requests `k+1` neighbours, locates the self-match row-wise (handles ties), and drops it deterministically. Both FAISS and numpy branches behave identically. Regression test: `test_get_knn_excludes_self_numpy_branch`.

### 9. `MetricRegistry` singleton has no reset
`metrics/registry.py:32-41`. Custom metrics registered in one test leak into the next.

**Fix:** Add `reset()` / `clear()` method, plus a `pytest` autouse fixture in conftest.

**Applied** — `MetricRegistry.reset()` keeps the three built-ins and drops everything else; `tests/conftest.py` autouse fixture calls it around every test.

---

## Design / API

### 10. `Comparison.__post_init__` mutates a frozen dataclass via `object.__setattr__`
`core/comparison.py:65`.

**Fix:** Make `severity` a `@property` computed from `global_metrics` + `thresholds`.

**Applied** — `severity` is now a `@property` on the frozen dataclass; no setattr-on-frozen hack. Default thresholds live as a class constant.

### 11. `LinearTransfer.fit` hardcodes 3 features
`transfer/function.py:89-90` slices `solution[:3]` / `solution[3]`. Brittle.

**Fix:** Drive off `X.shape[1]`.

**Applied** — both `LinearTransfer.fit` and `LogisticTransfer.fit` now slice by `X.shape[1]`.

### 12. `LogisticTransfer._extract_features` duplicates `LinearTransfer._extract_features`
`function.py:197-204` vs `:123-145`.

**Fix:** Pull up to base class or module function.

**Applied** — `extract_drift_features()` lives at module scope; the `_extract_features` method on the `TransferFunction` ABC calls it. Feature names live in the module-level `TRANSFER_FEATURE_NAMES` tuple, shared by `CalibrationProfile`.

### 13. `Snapshot.tower_names: tuple` and `embeddings: dict` lose generic info
`snapshot.py:36-37`. With `mypy strict = true` in pyproject, this weakens inference.

**Fix:** `tuple[str, ...]`, `dict[str, np.ndarray]`.

**Applied.**

### 14. Frozen `Snapshot` with mutable `dict` field
Implicit `__hash__` fails because `dict` is unhashable. Breaks if used as dict key.

**Fix:** `eq=False`, or implement `__hash__` from `checkpoint_hash`.

**Applied** — `@dataclass(frozen=True, eq=False)` + explicit `__hash__` from the integrity fields (model_id, checkpoint_hash, timestamp, anchor_set_version, tower_count, tower_names).

### 15. Integrations layer is scaffolding only
`integrations/__init__.py` doesn't export `ConsoleLogger`; W&B/MLflow are in `pyproject.toml` extras but have no implementation. README claims "MLOps ready."

**Fix:** Ship `WandbLogger` / `MLflowLogger`, or remove the claim until they exist.

**Applied**: (partial) — README claim toned down to "scaffolded, wire-up in progress". Shipping real W&B / MLflow loggers is a follow-up; tracked here for the next pass.

### 16. `CalibrationProfile.feature_names: list[str] = None`
`transfer/calibration.py:33`. Mutable default + wrong type.

**Fix:** `Optional[list[str]] = None` with `field(default=None)`, or `field(default_factory=lambda: [...])`.

**Applied** — `feature_names: list[str] = field(default_factory=lambda: list(TRANSFER_FEATURE_NAMES))`. Removed the `__post_init__` hack.

---

## Polish

### 17. README placeholders
`README.md:103` — `git clone https://github.com/yourusername/...`. `:123` cites placeholder URL. Line 134 has a duplicate `# semantic-sentry` header. References `LICENSE` and `CONTRIBUTING.md` files that may not exist.

**Applied** — both URLs now point at `github.com/abtonmoy/semantic-sentry`; duplicate header removed; `LICENSE` (Apache 2.0) added; `CONTRIBUTING.md` still TBD (low priority).

### 18. Examples mismatch README
README lists `clip_drift_detection.py`; file doesn't exist.

**Applied** — `clip_drift_detection.py` bullet removed from the Examples list.

### 19. Test layout is inconsistent
`tests/test_experiment_1.py` and `tests/test_experiment_1_smoke.py` sit at root alongside `unit/`, `integration/`, `stress/`. Move under `integration/`.

**Applied** — both files moved into `tests/integration/`.

### 20. No CI
No `.github/workflows/`, no pre-commit. With ruff + mypy + pytest already configured in pyproject, one workflow buys a lot.

**Applied** — `.github/workflows/test.yml` (ruff + mypy + pytest, py3.10 + py3.11), `.pre-commit-config.yaml` (ruff + ruff-format + standard hygiene hooks).
