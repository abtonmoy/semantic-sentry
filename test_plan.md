# SemanticSentry Stress Test Plan

## Executive Summary

This document outlines comprehensive stress testing for SemanticSentry to validate robustness, performance, and correctness under extreme conditions.

**Test Coverage Areas:**
- Metric robustness (CKA, NPS, Isotropy)
- Scale & performance
- Concurrency & thread safety
- Data quality edge cases
- Adversarial scenarios

---

## Phase 1: Metric Robustness Tests

### 1.1 CKA Stress Tests

| Test ID | Description | Input | Expected | Priority |
|---------|-------------|-------|----------|----------|
| CKA-001 | Identical matrices | Z0 = Z1 | CKA == 1.0 exactly | P0 |
| CKA-002 | Rank-deficient | Z has rank 1 | CKA > 0.95 vs self | P0 |
| CKA-003 | Orthogonal matrices | Z1 = Z0 @ Q (orthogonal Q) | CKA == 1.0 | P0 |
| CKA-004 | Very different | Random Z0, Z1 | CKA < 0.5 | P1 |
| CKA-005 | Zero matrix | Z = zeros | CKA == 0 or error | P1 |
| CKA-006 | Single sample | n=1 | Handle gracefully | P1 |
| CKA-007 | High condition number | ill-conditioned | Numerical stability | P2 |

### 1.2 NPS Stress Tests

| Test ID | Description | Input | Expected | Priority |
|---------|-------------|-------|----------|----------|
| NPS-001 | Self-comparison | Z0 = Z1 | NPS == 1.0 | P0 |
| NPS-002 | k > n | k=100, n=10 | Handle or error | P0 |
| NPS-003 | Identical embeddings | All rows equal | Defined behavior | P0 |
| NPS-004 | Random shuffle | Permuted Z1 | High NPS | P1 |
| NPS-005 | Small perturbation | Z1 = Z0 + ε | Monotonic drop | P1 |
| NPS-006 | Large n, small k | n=10k, k=5 | Performance | P1 |
| NPS-007 | Adversarial k-NN | Structured to break NPS | Robustness | P2 |

### 1.3 Isotropy Stress Tests

| Test ID | Description | Input | Expected | Priority |
|---------|-------------|-------|----------|----------|
| ISO-001 | Perfectly isotropic | Spherical Gaussian | ISO > 0.9 | P0 |
| ISO-002 | Rank-1 matrix | Single direction | ISO < 0.1 | P0 |
| ISO-003 | Square matrix | n=d | Stable computation | P1 |
| ISO-004 | Tall matrix | n << d | SVD handling | P1 |
| ISO-005 | Wide matrix | n >> d | SVD handling | P1 |

---

## Phase 2: Scale & Performance Tests

### 2.1 Anchor Set Scale Tests

| Test ID | Samples | Dimension | Memory Target | Time Target | Priority |
|---------|---------|-----------|---------------|-------------|----------|
| SCL-001 | 1,000 | 128 | < 100MB | < 1s | P0 |
| SCL-002 | 10,000 | 512 | < 500MB | < 5s | P0 |
| SCL-003 | 100,000 | 1024 | < 2GB | < 30s | P1 |
| SCL-004 | 1,000,000 | 768 | < 8GB | < 5min | P2 |

### 2.2 High-Dimensional Embeddings

| Test ID | Dimension | Strategy | Priority |
|---------|-----------|----------|----------|
| DIM-001 | 4,096 | Chunked computation | P1 |
| DIM-002 | 8,192 | Approximate CKA | P2 |
| DIM-003 | 16,384 | Dimensionality reduction | P2 |

### 2.3 Memory Profiling

- **Profile targets**: snapshot(), compare(), MetricRegistry.compute_all()
- **Tools**: tracemalloc, memory_profiler
- **Leak detection**: Create/destroy 1000 snapshots, check growth

---

## Phase 3: Concurrency & Thread Safety

### 3.1 MetricRegistry Tests

| Test ID | Scenario | Threads | Expected | Priority |
|---------|----------|---------|----------|----------|
| THR-001 | Concurrent register | 100 | No corruption | P0 |
| THR-002 | Concurrent compute | 100 | Consistent results | P0 |
| THR-003 | Register during compute | 50+50 | No deadlock | P0 |

### 3.2 DriftMonitor Tests

| Test ID | Scenario | Expected | Priority |
|---------|----------|----------|----------|
| THR-004 | Parallel snapshot() | Thread-safe | P1 |
| THR-005 | Parallel compare() | Thread-safe | P1 |
| THR-006 | Concurrent save/load | No corruption | P1 |

---

## Phase 4: Data Quality Edge Cases

### 4.1 Embedding Anomalies

| Test ID | Anomaly | Expected Behavior | Priority |
|---------|---------|-------------------|----------|
| DQA-001 | NaN values | Raise error | P0 |
| DQA-002 | Inf values | Raise error | P0 |
| DQA-003 | All zeros | Handle gracefully | P1 |
| DQA-004 | Very large values | Numerical stability | P1 |
| DQA-005 | Very small values | Numerical stability | P1 |

### 4.2 Anchor Set Edge Cases

| Test ID | Case | Expected | Priority |
|---------|------|----------|----------|
| DQA-006 | Empty set | Error | P0 |
| DQA-007 | Single sample | Handle | P0 |
| DQA-008 | All same label | Valid | P1 |
| DQA-009 | Unicode/emoji | Handle | P1 |
| DQA-010 | Very long text | Truncate/stream | P2 |

---

## Phase 5: Adversarial Scenarios

### 5.1 Drift Injection Attacks

| Test ID | Attack | Expected | Priority |
|---------|--------|----------|----------|
| ADV-001 | Rotation attack | CKA=1, NPS<1 | P1 |
| ADV-002 | Subspace attack | Detect in minor SVs | P2 |
| ADV-003 | Sparse perturbation | High NPS, low CKA | P2 |

### 5.2 Transfer Function Robustness

| Test ID | Scenario | Expected | Priority |
|---------|----------|----------|----------|
| ADV-004 | Outlier degradation | Robust fit | P1 |
| ADV-005 | Non-monotonic pattern | Handle gracefully | P2 |

### 5.3 Serialization Corruption

| Test ID | Corruption | Expected | Priority |
|---------|------------|----------|----------|
| ADV-006 | Truncated file | Error | P0 |
| ADV-007 | Modified hash | Integrity error | P0 |
| ADV-008 | Extra fields | Ignore/parse | P1 |

---

## Phase 6: Real-World Model Testing

**Goal**: Test with actual pre-trained models from HuggingFace Hub, not synthetic models.

### 6.1 Pre-trained Checkpoint Tests

| Test ID | Model | Scenario | Expected | Priority |
|---------|-------|----------|----------|----------|
| REAL-001 | bert-base-uncased | Load from HF Hub | Success | P0 |
| REAL-002 | bert-base vs sentiment-finetuned | Compare base vs fine-tuned | Detect drift | P0 |
| REAL-003 | all-MiniLM-L6-v2 | SentenceTransformer | Success | P0 |
| REAL-004 | CLIP ViT-B/32 | Vision-language | Success | P1 |
| REAL-005 | E5-large-v2 | Embedding model | Success | P1 |
| REAL-006 | GPT-2 | Generative model | Success | P2 |

### 6.2 Quantization Tests

| Test ID | Base | Quantized | Expected | Priority |
|---------|------|-----------|----------|----------|
| REAL-007 | FP32 | INT8 (bitsandbytes) | Detect drift | P0 |
| REAL-008 | FP32 | BF16 | Detect drift | P1 |
| REAL-009 | FP32 | QLoRA 4-bit | Detect drift | P1 |
| REAL-010 | FP16 | INT8 ONNX | Detect drift | P2 |

### 6.3 Version Evolution

| Test ID | From | To | Expected | Priority |
|---------|------|----|----------|----------|
| REAL-011 | v1.0 | v1.1 (patch) | Low drift | P1 |
| REAL-012 | base | large | High drift | P1 |
| REAL-013 | epoch-0 | epoch-10 | Increasing drift | P2 |

### 6.4 Cross-Platform Persistence

| Test ID | Scenario | Expected | Priority |
|---------|----------|----------|----------|
| REAL-014 | Save on CPU, load on GPU | Works | P1 |
| REAL-015 | Different PyTorch versions | Backward compat | P2 |
| REAL-016 | Different Python versions | Works | P2 |

### 6.5 Real Data Anchor Sets

| Test ID | Dataset | Size | Priority |
|---------|---------|------|----------|
| REAL-017 | IMDB reviews | 25k | P1 |
| REAL-018 | COCO captions | 5k | P1 |
| REAL-019 | Wiki passages | 100k | P2 |
| REAL-020 | Multilingual (XNLI) | 10k | P2 |

---

## Test Implementation Matrix

| Component | Unit Tests | Integration | Stress | Total |
|-----------|------------|-------------|--------|-------|
| CKA | 10 | 5 | 7 | 22 |
| NPS | 12 | 5 | 7 | 24 |
| Isotropy | 8 | 3 | 5 | 16 |
| Snapshot | 10 | 4 | 4 | 18 |
| Comparison | 11 | 4 | 3 | 18 |
| Registry | 13 | 3 | 3 | 19 |
| Adapter | 11 | 4 | 4 | 19 |
| Monitor | 0 | 6 | 6 | 12 |
| Transfer | 0 | 3 | 5 | 8 |
| **Total** | **75** | **37** | **44** | **156** |

---

## Success Criteria

### Performance Benchmarks
- **Small**: < 1000 samples, < 1s
- **Medium**: < 10k samples, < 10s
- **Large**: < 100k samples, < 60s
- **XLarge**: < 1M samples, < 5min

### Reliability Targets
- 100% pass rate for P0 tests
- 95% pass rate for P1 tests
- 80% pass rate for P2 tests

### Memory Limits
- No memory leaks (>1% growth after 1000 iterations)
- Graceful OOM handling with informative errors

---

## Execution Schedule

| Phase | Tests | Estimated Time | Owner |
|-------|-------|----------------|-------|
| Phase 1 | 42 tests | 2 days | Dev |
| Phase 2 | 10 tests | 2 days | Dev |
| Phase 3 | 16 tests | 1 day | Dev |
| Phase 4 | 20 tests | 1 day | Dev |
| Phase 5 | 15 tests | 1 day | Dev |
| Phase 6 | 20 tests | 3 days | Dev |
| **Total** | **123 tests** | **10 days** | - |

---

## Notes

- Priority levels: P0 (critical), P1 (important), P2 (nice-to-have)
- All tests should be deterministic (use fixed random seeds)
- Include hypothesis-based property testing where applicable
- Document expected failures for known limitations
