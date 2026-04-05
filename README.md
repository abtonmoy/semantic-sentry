# SemanticSentry

Universal semantic drift detection for any embedding space.

[![PyPI version](https://badge.fury.io/py/semantic-sentry.svg)](https://badge.fury.io/py/semantic-sentry)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

SemanticSentry detects and quantifies semantic drift in continuously updated embedding models. It monitors how model updates (fine-tuning, quantization, LoRA, etc.) geometrically transform embedding spaces and predicts downstream performance degradation—without requiring labeled evaluation data.

### Key Features

- 🎯 **Universal**: Works with any embedding model (BERT, CLIP, sentence transformers, custom models)
- 📊 **Multi-metric**: CKA (global), NPS (local), Isotropy (spectral)
- 🔮 **Predictive**: Transfer functions predict downstream degradation
- 🏗️ **Multi-tower**: Supports dual-encoder and multi-modal models
- 🚀 **Efficient**: FAISS-accelerated neighborhood search
- 📝 **MLOps ready**: Weights & Biases, MLflow integrations

## Quick Start

### Installation

```bash
# Basic installation
pip install semantic-sentry

# With optional dependencies
pip install semantic-sentry[clip]           # CLIP support
pip install semantic-sentry[sentence-transformers]  # SentenceTransformers
pip install semantic-sentry[onnx]           # ONNX support
pip install semantic-sentry[all]            # Everything
```

### Basic Usage

```python
from semantic_sentry import DriftMonitor, AnchorSet
from semantic_sentry.adapters.custom import CustomAdapter
import torch

# Create anchor set
anchor_set = AnchorSet(
    inputs=["example text 1", "example text 2", ...],
    labels=("label_1", "label_2", ...),
)

# Create adapter for your model
adapter = CustomAdapter(encode_fn=model.encode, tower_count=1)

# Capture snapshots
monitor = DriftMonitor()
snapshot_v0 = monitor.snapshot(model_v0, anchor_set, adapter=adapter)
snapshot_v1 = monitor.snapshot(model_v1, anchor_set, adapter=adapter)

# Compare and detect drift
comparison = monitor.compare(snapshot_v0, snapshot_v1)
print(f"Drift severity: {comparison.severity}")
print(f"CKA: {comparison.global_metrics['cka']:.4f}")
print(f"NPS: {comparison.global_metrics['nps']:.4f}")
```

## Architecture

SemanticSentry follows a 5-layer architecture:

1. **Encoder Layer**: Model-specific adapters (HuggingFace, CLIP, SentenceTransformers, ONNX, Custom)
2. **Snapshot Layer**: Frozen capture of embedding states
3. **Metrics Layer**: CKA, NPS, Isotropy with registry pattern
4. **Transfer Layer**: Linear transfer functions, calibration profiles
5. **Integration Layer**: W&B, MLflow, webhooks

## Supported Metrics

| Metric | Description | Range | Property |
|--------|-------------|-------|----------|
| CKA | Centered Kernel Alignment | [0, 1] | Global structural similarity |
| NPS | Neighborhood Preservation Score | [0, 1] | Local neighborhood retention |
| Isotropy Δ | Spectral geometry change | [-1, 1] | Anisotropy shift |

## Alert Severity Levels

| Severity | CKA | NPS | Action |
|----------|-----|-----|--------|
| Low | > 0.98 | > 0.95 | ✓ Stable |
| Medium | 0.90-0.98 | 0.85-0.95 | ⚠ Monitor |
| High | 0.80-0.90 | 0.70-0.85 | ⚠ Evaluate |
| Critical | < 0.80 | < 0.70 | ✗ Retrain |

## Examples

See `examples/` directory:

- `quickstart.py`: Minimal usage example
- `text_encoder_monitoring.py`: BERT/E5-style monitoring
- `clip_drift_detection.py`: Vision-language model monitoring

## Development

```bash
# Clone repository
git clone https://github.com/yourusername/semantic-sentry
cd semantic-sentry

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/ --cov=semantic_sentry --cov-report=html
```

## Citation

```bibtex
@software{semantic_sentry,
  title = {SemanticSentry: Universal Semantic Drift Detection},
  author = {Abdul Basit Tonmoy},
  year = {2025},
  url = {https://github.com/yourusername/semantic-sentry}
}
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
# semantic-sentry
