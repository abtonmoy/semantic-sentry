# SemanticSentry

Drift detection for embedding spaces. Capture a snapshot of an
embedding model's output, snapshot it again later, and get a small,
auditable set of numbers describing what changed.

[![PyPI](https://img.shields.io/pypi/v/semantic-sentry.svg)](https://pypi.org/project/semantic-sentry/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-194%20passing-brightgreen)](tests/unit/)

---

## Why

When you fine-tune, quantise, distil, or swap a base model, the
embedding space changes. SemanticSentry quantifies the change with
metrics that are well-defined, scale-invariant, and reproducible across
runs:

- **Geometric drift** — how much did the representation move?
- **Local structure** — did neighbourhoods reorganise?
- **Ranking behaviour** — would a retrieval index return different
  results?
- **Spectral shape** — did the singular-value spectrum shift?
- **Temporal dynamics** — is drift accelerating or has it plateaued?

The library gives you each of those as a single function call, plus
a snapshot/compare orchestration layer for CI/CD and monitoring
workflows.

## Installation

```bash
pip install semantic-sentry
```

Optional extras:

```bash
pip install "semantic-sentry[clip]"                    # CLIP adapter
pip install "semantic-sentry[sentence-transformers]"   # SentenceTransformer adapter
pip install "semantic-sentry[onnx]"                    # ONNX adapter
pip install "semantic-sentry[bench]"                   # BEIR + MTEB benchmark suites
pip install "semantic-sentry[wandb,mlflow]"            # logging integrations
pip install "semantic-sentry[all]"                     # everything
```

Requires Python 3.10+.

## Quick start

```python
from semantic_sentry import DriftMonitor, AnchorSet
from semantic_sentry.adapters.custom import CustomAdapter

# A fixed anchor set the metrics are computed against.
anchor = AnchorSet(
    inputs=["example 1", "example 2", "example 3", ...],
    labels=("a", "b", "a", ...),
)

# Wrap your model behind an adapter.
adapter = CustomAdapter(encode_fn=model.encode, tower_count=1)

monitor = DriftMonitor()
before = monitor.snapshot(model_v0, anchor, adapter=adapter)
after  = monitor.snapshot(model_v1, anchor, adapter=adapter)

comparison = monitor.compare(before, after)
print(f"CKA: {comparison.global_metrics['cka']:.4f}")
print(f"NPS: {comparison.global_metrics['nps']:.4f}")
print(f"Severity: {comparison.severity}")
```

### Calling metrics directly

If you don't need the orchestration layer, every metric is also a
plain function over numpy arrays:

```python
from semantic_sentry.metrics.cka import linear_cka
from semantic_sentry.metrics.nps import nps
from semantic_sentry.metrics.local_structure import trustworthiness, continuity

Z_old = old_model.encode(anchor_texts)   # shape (n, d)
Z_new = new_model.encode(anchor_texts)   # shape (n, d)

cka  = linear_cka(Z_old, Z_new)
nps_ = nps(Z_old, Z_new, k=10)
trust = trustworthiness(Z_old, Z_new, k=10)
cont  = continuity(Z_old, Z_new, k=10)
```

All metric functions are deterministic, rotation- and
permutation-invariant where mathematically appropriate, and unit-tested.

## Features

### Adapters

Pre-built encoder adapters for common model types, plus a
`CustomAdapter` for anything else.

| Adapter | Module |
|---|---|
| HuggingFace `PreTrainedModel` + tokenizer | `adapters/huggingface.py` |
| OpenCLIP / OpenAI CLIP | `adapters/clip.py` |
| `sentence_transformers.SentenceTransformer` | `adapters/sentence_transformer.py` |
| ONNX `InferenceSession` | `adapters/onnx_adapter.py` |
| Arbitrary callable | `adapters/custom.py` |

Multi-tower models (CLIP-style image+text encoders) are first-class.

### Metrics

#### Geometric (registered by default)

| Metric | Range | Property |
|---|---|---|
| Linear CKA (Kornblith et al., 2019) | [0, 1] | Global structural similarity. Rotation and permutation invariant. |
| NPS at k (with FAISS) | [0, 1] | Mean per-point overlap of top-k neighbours between two snapshots. |
| Isotropy Δ | [-1, 1] | Change in spectral anisotropy. |

#### Local structure (opt-in)

| Metric | Range | Property |
|---|---|---|
| Trustworthiness | [0, 1] | Penalises false neighbours introduced in Z₁. |
| Continuity | [0, 1] | Penalises true neighbours from Z₀ that fell out of Z₁'s top-k. |

`continuity(A, B) == trustworthiness(B, A)`. Register both with
`register_local_structure_metrics()` to get them in `compute_all`
output.

#### Behavioural / ranking (opt-in, four-input)

These need a query-side and a document-side matrix
(`fn(Z0_Q, Z1_Q, D0, D1)`). Use `AnchorSet.partition()` for a Q/D split,
then `compare(d_snapshot_v0=…, d_snapshot_v1=…)`. Register with
`register_behavioral_metrics()`.

| Metric | Range | Property |
|---|---|---|
| `score_distribution_jsd` | [0, 1] | JSD between baseline and updated cosine-sim distributions over Q×D pairs. |
| `mean_abs_score_delta` | [0, 2] | Mean pointwise score change over Q×D pairs. |
| `per_query_rbo` (Webber, 2010) | [0, 1] | Top-weighted rank-biased overlap; default `p = 0.9`. |
| `per_query_kendall_tau` | [-1, 1] | Mean Kendall τ on the full D-side ranking. |
| `self_retrieval_topk_at_{k}` | [0, 1] | Fraction of queries whose top-k document set is unchanged. |

#### Temporal wrappers

`metrics/temporal.py` wraps any registered metric over a checkpoint
trajectory.

| Function | Returns | Property |
|---|---|---|
| `velocity(snapshots, times, metric_name)` | `np.ndarray (T,)` | dM/dt via central differences. |
| `acceleration(snapshots, times, metric_name)` | `np.ndarray (T,)` | d²M/dt². |
| `plateau(snapshots, times, metric_name, eps, delta, k)` | `np.ndarray (T,)` bool | True where `|v| < eps` and `|a| < delta` for `k` consecutive checkpoints. |

### Calibrating severity thresholds

The default severity bands (`low / medium / high / critical`) use
fixed CKA and NPS thresholds. They're convenient defaults but they
treat every model family the same, which is rarely what you want —
CLIP wobbles at σ ≈ 0.018 across seeds while a sentence-transformer
might wobble at σ ≈ 0.002. `calibrate_thresholds()` derives
per-family bands from a small set of no-drift reference snapshots:

```python
from semantic_sentry.core.calibration import calibrate_thresholds

# `reference_snapshots`: 3+ snapshots from "no real drift" pairs.
cal = calibrate_thresholds(reference_snapshots, metrics=("nps", "cka"))

cmp = monitor.compare(snapshot_v0, snapshot_v1, calibration=cal)
print(cmp.severity)   # uses cal.thresholds rather than the static defaults
```

### Snapshot persistence

```python
snapshot.save("snapshot_v0")
loaded = Snapshot.load("snapshot_v0")
assert loaded == snapshot
```

Saved as `metadata.json` plus one `{tower_name}.safetensors` per tower.

## Default severity buckets

If you don't supply a calibration, the defaults below apply.
Treat them as triage hints, not performance predictions — they're
purely geometric and won't tell you whether a downstream task will
degrade.

| Severity | CKA | NPS | Action |
|---|---|---|---|
| Low | > 0.98 | > 0.95 | Likely stable; verify with one downstream eval. |
| Medium | 0.90–0.98 | 0.85–0.95 | Monitor; investigate if downstream is critical. |
| High | 0.80–0.90 | 0.70–0.85 | Run downstream evals; do not assume preservation. |
| Critical | < 0.80 | < 0.70 | Re-evaluate and likely retrain. |

## Monitoring during training

`DriftMonitor.track()` collapses the snapshot/compare dance into one call
against a rolling baseline. The first call records the baseline; every
later call returns a `Comparison`:

```python
monitor = DriftMonitor()
for step, ckpt in enumerate(checkpoints):
    cmp = monitor.track(ckpt, anchor_set, step=step, adapter=adapter)
    if cmp and cmp.severity is AlertSeverity.CRITICAL:
        break  # geometry moved too far — stop and investigate
```

The monitor's constructor configures how tracking behaves:

```python
monitor = DriftMonitor(
    baseline_mode="previous",   # "fixed" (vs first ckpt) or "previous" (sliding window)
    track_temporal=True,        # attach velocity / acceleration / plateau signals
    plateau_k=3,                # plateau = settled for k consecutive checkpoints
    async_mode=True,            # offload metric compute + logging to a worker thread
)
```

- **`baseline_mode`** — `"fixed"` measures drift since monitoring started;
  `"previous"` measures step-to-step change against the prior checkpoint.
- **`track_temporal`** — each result carries
  `comparison.metadata["temporal"]` with `velocity`, `acceleration`, and a
  boolean `plateau` (the early-stopping signal: geometry has settled).
- **`async_mode`** — embeddings are captured synchronously (before the
  weights move on) but metrics + logging run off the training thread;
  `track()` returns a `Future`. Call `monitor.drain()` / `monitor.close()`
  to flush, and read `monitor.last_result` for the latest completed result.
- **`keep_on_device=True`** (per-call) computes the built-in metrics with the
  torch backend directly from the encoded tensors, skipping the numpy/snapshot
  round-trip — cheapest when tracking every N steps on GPU.

### Per-step tracking

The callbacks default to tracking at the trainer's eval/save events. For
fine-grained, per-step cadence, set `every_n_steps` (the callback then hooks
`on_step_end` / `on_train_batch_end`):

```python
monitor = DriftMonitor(async_mode=True, max_inflight=1, keep_on_device=False)
cb = SemanticSentryCallback(small_anchor_set, adapter=adapter, monitor=monitor,
                            every_n_steps=50)   # 1 = literally every step
```

Each measurement re-encodes the anchor set through the current weights — a
forward pass that runs on the training thread (it must observe the live
weights). To keep that off the critical path:

- **`every_n_steps`** sets the stride — the main cost lever.
- a **small anchor set** (encode cost scales with anchor count).
- **`async_mode` + `max_inflight=1`** run the metric math + logging on a
  worker and *drop* a measurement when the worker is still busy, so a slow
  step can never throttle training or backlog the queue.
- **`probe_eval_mode=True`** (default) flips the model to `eval()` for the
  probe and restores train mode — without it, per-step measurements pick up
  dropout/batchnorm noise.

Per-step deltas are small and noisy; lean on the temporal velocity/plateau
signals (`track_temporal=True`) rather than reading individual points.

### Framework callbacks

Drop a callback into your trainer and drift is tracked automatically, with
optional logging to W&B / MLflow:

```python
from semantic_sentry.adapters.huggingface import HuggingFaceAdapter
from semantic_sentry.integrations import SemanticSentryCallback, WandbLogger

adapter = HuggingFaceAdapter(model, tokenizer)
trainer = Trainer(
    ...,
    callbacks=[SemanticSentryCallback(anchor_set, adapter=adapter,
                                      logger=WandbLogger())],
)
```

To stop training automatically once the embedding geometry settles, pass a
temporal monitor and `stop_on_plateau=True` (sets `control.should_training_stop`):

```python
monitor = DriftMonitor(track_temporal=True)
cb = SemanticSentryCallback(anchor_set, adapter=adapter, monitor=monitor,
                            stop_on_plateau=True)
```

A `SemanticSentryLightningCallback` (same options, sets `trainer.should_stop`)
is available for PyTorch Lightning.

### Downstream proxies

When anchors are labelled, hand `track()` (or `compare()`) a list of
evaluators to get measured task deltas reported under
`comparison.metadata["downstream"]`:

```python
from semantic_sentry import RetrievalEvaluator
cmp = monitor.track(model, anchor_set, evaluators=[RetrievalEvaluator(k=10)])
print(cmp.metadata["downstream"])  # {"RetrievalEvaluator": -0.043}
```

## Command line

The CLI operates on snapshots already saved with `Snapshot.save()`, so it
runs anywhere (no model loading):

```bash
semantic-sentry info     snapshots/baseline
semantic-sentry compare  snapshots/baseline snapshots/candidate
# Exit code 1 if any threshold is violated — wire straight into CI:
semantic-sentry gate     snapshots/baseline snapshots/candidate \
    --fail-under cka=0.90,nps=0.85 --fail-over isotropy_delta=0.05
```

A composite GitHub Action wraps the gate (see `action.yml` and
`.github/workflows/drift-gate.example.yml`):

```yaml
- uses: abtonmoy/semantic-sentry@main
  with:
    baseline: snapshots/baseline
    candidate: snapshots/candidate
    fail-under: cka=0.90,nps=0.85
```

## Architecture

```
src/semantic_sentry/
├── adapters/         huggingface / clip / sentence_transformers / onnx / custom
├── core/
│   ├── snapshot.py     frozen embedding-state capture
│   ├── monitor.py      DriftMonitor orchestration + ClassificationContext
│   ├── comparison.py   Comparison dataclass + severity property
│   ├── classification.py
│   └── calibration.py  SeverityCalibration / calibrate_thresholds
├── metrics/
│   ├── cka.py             Kornblith linear CKA via centered-Gram HSIC
│   ├── nps.py             FAISS k-NN, self-exclusion-correct
│   ├── isotropy.py        spectral isotropy delta
│   ├── registry.py        MetricRegistry, register_at_k, MetricEntry.params
│   ├── behavioral.py      Q/D ranking metrics + BehavioralMetricRegistry
│   ├── local_structure.py trustworthiness, continuity
│   └── temporal.py        velocity, acceleration, plateau
├── probes/           AnchorSet (with .partition())
├── evaluation/       downstream-metric harnesses
├── transfer/         linear + MLE-logistic transfer fns, calibration profiles
├── integrations/     wandb / mlflow / webhook scaffolding
└── exceptions.py
```

## Examples

See `examples/`:

- `quickstart.py` — minimal end-to-end usage.
- `text_encoder_monitoring.py` — BERT / sentence-transformer drift
  detection.

## Development

```bash
git clone https://github.com/abtonmoy/semantic-sentry
cd semantic-sentry
pip install -e ".[dev]"

pytest tests/unit/ -q           # 194 unit tests
pytest tests/unit/ tests/stress/ -m "not slow"   # + 72-test stress suite
pytest tests/ --cov=semantic_sentry              # with coverage
```

The repo enforces `ruff` for lint and `mypy` for typing; both run in
CI on Python 3.10 and 3.11.

## Contributing

Contributions welcome — issues and PRs at
[github.com/abtonmoy/semantic-sentry](https://github.com/abtonmoy/semantic-sentry).
See [CONTRIBUTING.md](CONTRIBUTING.md) if present.

## Citation

```bibtex
@software{semantic_sentry,
  title  = {SemanticSentry: Drift Detection for Embedding Spaces},
  author = {Abdul Basit Tonmoy},
  year   = {2026},
  url    = {https://github.com/abtonmoy/semantic-sentry}
}
```

## License

[Apache 2.0](LICENSE).
