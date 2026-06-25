# Changelog

All notable changes to SemanticSentry. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-06-25

### Breaking — `LinearTransfer` is signed by default; clip is opt-in

`LinearTransfer.predict()` now returns the raw OLS prediction (`w · x + b`)
unmodified by default. To recover the v0.1.x clamp-to-`[0, 1]` behavior,
pass `clip=True` at construction time. When `clip=True` is passed, `fit()`
also raises `ValueError` on any negative degradation so the two halves of
the contract stay consistent.

#### What changed

| | v0.1.x | v0.2.0 default (signed) | v0.2.0 opt-in (`clip=True`) |
|---|---|---|---|
| `fit([..., -0.04, ...])` | returned silently | accepted, returns signed predictions | raises `ValueError` |
| `predict()` return domain | `[0, 1]` (silently clamped) | signed real number | `[0, 1]` |
| Negative predictions | silently collapsed to `0.0` | preserved | rejected at fit-time |

#### Migration

If your calibration targets can be negative (the candidate sometimes
*improves* the downstream metric — e.g. `ΔnDCG` with `candidate > baseline`):

```python
# v0.1.x: silently broken — predict() returned 0 for every negative-target
# regime, destroying both magnitude and rank.
transfer = LinearTransfer()
transfer.fit(comparisons, [-0.04, -0.06, -0.02, ...])
transfer.predict(c)  # always 0.0 — bug

# v0.2.0 default: signed predictions, works correctly out of the box.
transfer = LinearTransfer()
transfer.fit(comparisons, [-0.04, -0.06, -0.02, ...])
transfer.predict(c)  # signed, ≈ -0.04 — what the regression actually fit
```

If you were relying on v0.1.x's clamp behavior and your degradations are
genuinely non-negative magnitudes (e.g. you pre-processed with
`max(0, baseline - candidate)`), opt in to the same behavior with one
keyword:

```python
# v0.2.0: explicit opt-in to the legacy non-negative-degradation contract.
transfer = LinearTransfer(clip=True)
transfer.fit(comparisons, non_negative_degradations)  # raises if any < 0
transfer.predict(c)  # in [0, 1]
```

#### Why the default flipped

In v0.1.x, `predict()` clamped its output to `[0, 1]` while `fit()` accepted
arbitrarily-signed targets via `np.linalg.lstsq`. When calibration targets
were negative — the common case for any downstream metric that can improve
under fine-tuning (`ΔnDCG`, `ΔaccuracyError`, `ΔF1` evaluated as
`baseline − candidate`) — every negative raw prediction collapsed to `0.0`
inside `predict()`. The fit was correct; the silent clamp at inference
destroyed magnitude *and* rank information for the entire regime.

The cleanest fix surfaced by reviewing the pilot data was to remove the
implicit assumption that "degradation" means "non-negative magnitude" and
make the unsigned-clamp behavior an explicit opt-in. The default now
matches the natural shape of an OLS regression: signed in, signed out.

#### Discovery context

Surfaced by the paper pilot on 2026-06-24, where C2 (proper-contrastive
fine-tuning on E5) consistently improved retrieval, producing all-negative
`ΔnDCG@10` targets. The pilot's LOSO transfer fits superficially looked
catastrophic (R² ≈ −10 across 6 folds) — but reproducing the fits offline
without the clamp gave R² ≈ +0.77 on the same data. Every C2 prediction
had been silently flattened to `0.0`.

### CalibrationProfile — `clip` is now persisted

`CalibrationProfile` gains a `clip: bool = False` field so the contract
round-trips through save/load without drift. Prior to this release,
`CalibrationProfile.to_transfer_function()` unconditionally constructed
`LinearTransfer()` (default args), which would have silently downgraded
any `clip=True` transfer to `clip=False` across a serialization cycle —
making the "clamp output" and "reject negative input" halves of the
contract effectively independent in any pipeline that ships profiles to
disk and reloads them later.

#### JSON format change

Saved profile JSON now contains an additional key:

```json
{
  "profile_name": "...",
  "model_family": "...",
  "weights": [...],
  "bias": ...,
  "r_squared": ...,
  "n_samples": ...,
  "feature_names": [...],
  "clip": false
}
```

`CalibrationProfile.load()` accepts profiles with or without the `clip`
key. Profiles written by v0.1.0 (no `clip` key) load as `clip=False`,
which is the new signed default — under v0.1.0's implementation those
profiles were silently clamped at predict-time regardless of fit, so
loading as the signed default is a behavioral upgrade for legacy
profiles, not a regression. If you need the old clamped behavior on a
v0.1.0 profile, edit the JSON to add `"clip": true` before loading.

#### What this prevents

`LinearTransfer(clip=True)` round-tripped through CalibrationProfile now
yields another `LinearTransfer(clip=True)` instance — both halves of the
contract (predict-clamp and fit-reject-negatives) ride on the single
`_clip` flag and cannot drift apart across the serialization boundary.

### Internal

  - `LinearTransfer.__init__` gains keyword-only `clip: bool = False`.
  - `LinearTransfer.fit()` only validates non-negative targets when
    `clip=True`. Default fit accepts mixed-sign degradations without error.
  - `LinearTransfer.predict()` returns signed `float` by default; clamps to
    `[0, 1]` only when `clip=True`.
  - `CalibrationProfile.from_transfer_function()` now reads
    `transfer._clip`; `to_transfer_function()` now passes it back to
    `LinearTransfer(clip=...)`; `save()` writes it; `load()` reads it with
    a default of `False` for backward compat.
  - Test `test_trf_prediction_clamped` retired; replaced by:
    - `test_trf_default_accepts_mixed_sign_degradations` (verifies signed
      default accepts all-negative and mixed-sign calibration data and
      returns unbounded predictions),
    - `test_trf_clip_opt_in_clamps_output` (verifies `clip=True` clamps
      predictions and rejects negative targets at fit-time),
    - `test_trf_clip_round_trips_through_calibration_profile` (verifies
      `clip=True` survives save/load and `to_transfer_function()` so the
      two halves of the contract cannot silently desync across
      serialization; also pins the v0.1.0 backward-compat default).
  - `LogisticTransfer` is unchanged. Its `degradation_threshold` already
    handles negative deltas correctly by labeling them `0`.

## [0.1.0] — 2026-04-16

Initial release. Three drift metrics (CKA, NPS, isotropy_delta), five
encoder adapters (HuggingFace, CLIP, SentenceTransformer, ONNX, custom),
transfer layer (Linear + Logistic), integration layer (W&B, MLflow,
ConsoleLogger).
