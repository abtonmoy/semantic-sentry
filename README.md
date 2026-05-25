# SemanticSentry

Universal semantic drift detection for embedding spaces.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

SemanticSentry is a small library for capturing snapshots of an embedding
space and computing standard geometric-drift metrics — linear CKA, NPS,
isotropy Δ — between snapshots. It is the metric-and-snapshot core that
the matched-magnitude dissociation paper (companion `../paper/` and
`../experiments/`) is built on.

### What the metrics do — and what they don't

Geometric drift metrics tell you **whether the embedding space moved**.
They do **not** tell you whether the downstream task will degrade,
improve, or stay the same. The companion paper establishes empirically
that on E5-base-v2 a contrastive LoRA at CKA 0.88 / NPS 0.60 preserves
MS MARCO retrieval, while MLM LoRA at CKA 0.81 / NPS 0.48 — barely more
geometric drift — destroys it. The relationship between drift magnitude
and downstream impact depends on the *direction* of drift relative to
the evaluation task. See `../experiments/figures/fig1_dissociation_5way.png`
for the 5-way visual summary and `../paper/sections/04-dissociation.tex`
Table 1 for the underlying numbers.

The metrics in this library are therefore useful as **change detectors**
(did something move?) and as **dissociation diagnostics** (is the
geometric drift consistent with the downstream impact, or is the
downstream impact larger / smaller than the drift would predict?) — not
as a one-number health score.

### Key features

- **Universal**: Works with any embedding model (BERT, CLIP,
  sentence-transformers, custom models).
- **Metrics**: Linear CKA (Kornblith et al. 2019), NPS at k-NN with
  FAISS, Isotropy Δ from singular-value spectra. All anchor-parameterised
  and unit-tested.
- **Snapshot/compare API**: Freeze an embedding-state, diff two
  snapshots, get per-metric numbers + severity heuristics.
- **Multi-tower**: Supports dual-encoder and multi-modal models.
- **Efficient**: FAISS-accelerated neighborhood search; metrics run
  in seconds on 1 000-passage anchors with 768-d embeddings.
- **Drops into your training loop**: a one-line `monitor.track(model,
  anchors, step=...)` plus ready-made `transformers.Trainer` and PyTorch
  Lightning callbacks — drift gets measured against a rolling baseline at
  every eval, with no changes to your training code.
- **Integrations**: `WandbLogger` and `MLflowLogger` push every metric next
  to your loss curves; `ConsoleLogger` for local runs. All under
  `src/semantic_sentry/integrations/`.
- **CLI + CI gate**: `semantic-sentry compare` / `gate` operate on saved
  snapshots and return a nonzero exit code when drift crosses your
  thresholds. Ships with a composite GitHub Action (`action.yml`).
- **Downstream proxies**: optional `RetrievalEvaluator` / `ClassificationEvaluator`
  report measured task deltas (MRR, kNN accuracy) alongside the geometric
  metrics when your anchors are labelled.

## Quick start

### Installation

```bash
# Basic installation
pip install semantic-sentry

# With optional dependencies
pip install semantic-sentry[clip] # CLIP support
pip install semantic-sentry[sentence-transformers] # SentenceTransformers
pip install semantic-sentry[onnx] # ONNX support
pip install semantic-sentry[all] # Everything
```

For development, clone and install editable:

```bash
git clone https://github.com/abtonmoy/semantic-sentry
cd semantic-sentry
pip install -e ".[dev]"
pytest tests/unit/ -q # 120 tests pass
```

### Basic usage

```python
from semantic_sentry import DriftMonitor, AnchorSet
from semantic_sentry.adapters.custom import CustomAdapter

# Fixed anchor set the metrics will be computed against
anchor_set = AnchorSet(
    inputs=["example text 1", "example text 2", ...],
    labels=("label_1", "label_2", ...),
)

# Adapter wraps your model's encode function
adapter = CustomAdapter(encode_fn=model.encode, tower_count=1)

# Snapshot before + after
monitor = DriftMonitor()
snapshot_v0 = monitor.snapshot(model_v0, anchor_set, adapter=adapter)
snapshot_v1 = monitor.snapshot(model_v1, anchor_set, adapter=adapter)

# Diff
comparison = monitor.compare(snapshot_v0, snapshot_v1)
print(f"CKA: {comparison.global_metrics['cka']:.4f}")
print(f"NPS: {comparison.global_metrics['nps']:.4f}")
print(f"Heuristic severity: {comparison.severity}") # interpret with the caveat above
```

You can also call the metric functions directly on numpy arrays:

```python
from semantic_sentry.metrics.cka import linear_cka
from semantic_sentry.metrics.nps import nps
import numpy as np

Z_base = base_model.encode(anchor_texts) # shape (n, d_base)
Z_new = new_model.encode(anchor_texts) # shape (n, d_new), n must match
cka_score = linear_cka(Z_base, Z_new)
nps_score = nps(Z_base, Z_new, k=10)
```

Both `linear_cka` and `nps` are rotation- and permutation-invariant
(verified in `tests/unit/test_cka.py` and `tests/unit/test_nps.py`).

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

SemanticSentry follows a 5-layer architecture:

1. **Encoder layer** — model-specific adapters (HuggingFace, CLIP,
   SentenceTransformers, ONNX, Custom).
2. **Snapshot layer** — frozen capture of embedding states + the anchor
   set they were computed against.
3. **Metrics layer** — CKA / NPS / isotropy Δ with a registry pattern
   for plugging in additional metrics.
4. **Transfer layer** — linear transfer functions and calibration
   profiles (used by the paper's downstream-prediction work).
5. **Integration layer** — W&B, MLflow, webhook adapters.

Source layout:

```
src/semantic_sentry/
├── adapters/ huggingface / clip / sentence_transformers / onnx / custom
├── core/ anchor sets, snapshots, monitor, comparison
├── metrics/ cka.py (Kornblith linear), nps.py (FAISS k-NN), isotropy.py
├── probes/ classification / retrieval probes
├── evaluation/ downstream-metric harnesses
├── transfer/ linear transfer fitting + calibration
├── integrations/ wandb, mlflow, webhooks
└── exceptions.py
```

## Metrics

| Metric | Implementation | Range | Property |
|---|---|---|---|
| Linear CKA | `metrics/cka.py` — centered-Gram HSIC form (Kornblith 2019) | [0, 1] | Global structural similarity. Rotation- and permutation-invariant. |
| NPS @ k | `metrics/nps.py` — FAISS `IndexFlatIP` on L2-normalised embeddings | [0, 1] | Mean per-point overlap of top-k neighbours between two snapshots. |
| Isotropy Δ | `metrics/isotropy.py` — singular-value spectrum gap | [-1, 1] | Change in spectral anisotropy. |

All three metrics are pure functions of `(Z_base, Z_new, [k])` and treat
the anchor as opaque, so the same code computes metrics against any
anchor set without modification.

## Heuristic severity buckets

Severity bands below are **heuristic** — useful for triage, not for
performance prediction. The matched-magnitude paper (see `../paper/`)
shows these thresholds are not reliable predictors of downstream
impact; e.g. a "High" CKA band (0.80–0.90) can correspond to either
preserved retrieval (contrastive LoRA) or 13 pp damage (MLM LoRA).

| Severity | CKA | NPS | Triage action |
|---|---|---|---|
| Low | > 0.98 | > 0.95 | Likely stable; verify with one downstream eval |
| Medium | 0.90–0.98 | 0.85–0.95 | Monitor; investigate if downstream eval is task-critical |
| High | 0.80–0.90 | 0.70–0.85 | Run downstream evals; do not assume preservation |
| Critical | < 0.80 | < 0.70 | Re-evaluate and likely retrain |

For a more principled diagnostic, see the matched-magnitude protocol in
`../experiments/figures/fig1_dissociation_5way.png` and §3 of the
companion paper.

## Examples

See `examples/`:

- `quickstart.py` — minimal usage example
- `text_encoder_monitoring.py` — BERT / E5-style monitoring

## Reproducing the matched-magnitude paper

The companion `../experiments/` directory holds every training run,
evaluation, and figure in the paper. See `../experiments/README.md` for
the full directory map and reproduction commands. Key scripts that use
this library:

- `../experiments/e5/e5_lora_finetune.py` — contrastive LoRA trainer that
  calls `compute_drift_metrics()` (NPS + CKA + isotropy Δ) at every
  checkpoint.
- `../experiments/methodology/anchor_robustness/anchor_robustness.py` —
  computes NPS + CKA for every Table 1 condition under three different
  anchor sets (recompute only, no retraining).

## Development

```bash
# Run the unit tests (120 tests — including invariance checks for CKA / NPS)
pytest tests/unit/ -q

# With coverage
pytest tests/ --cov=semantic_sentry --cov-report=html
```

## Citation

```bibtex
@software{semantic_sentry,
  title = {SemanticSentry: Universal Semantic Drift Detection},
  author = {Abdul Basit Tonmoy},
  year = {2026},
  url = {https://github.com/abtonmoy/semantic-sentry}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md)
for guidelines.
