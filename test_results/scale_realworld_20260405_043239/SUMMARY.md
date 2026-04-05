# Scale and Real-World Test Results Summary

**Job ID:** scale_realworld_20260405_043239  
**Date:** 2026-04-05T04:32:39Z  
**Status:** Completed

## Test Suite Overview

This job executed scale and real-world model tests:
- **Scale Tests:** Performance benchmarks at various dataset sizes
- **Real-World Tests:** Integration with actual HuggingFace models

## Results Summary

### Scale Tests

| Test | Status | Notes |
|------|--------|-------|
| test_scl_metric_performance[1000-128-1.0] | ✅ PASSED | <1 second for n=1000 |
| test_scl_metric_performance[10000-512-5.0] | ✅ PASSED | <5 seconds for n=10K |
| test_scl_metric_performance[100000-1024-30.0] | ⏱️ TIMEOUT | 30s limit exceeded |
| test_memory_leak_snapshot_cycle | ❌ FAILED | Memory grew 27,526x |
| test_scl_cka_scaling | ✅ PASSED | CKA scaling test passed |
| test_scl_nps_scaling | ✅ PASSED | NPS scaling test passed |

### Real-World Tests

| Test | Status | Notes |
|------|--------|-------|
| test_real_001_bert_base | ❌ FAILED | Bug in HuggingFaceAdapter (fixed after test) |
| test_real_003_sentence_transformer | ❌ FAILED | sentence_transformers not installed |
| test_real_012_different_dims_raises | ✅ PASSED | Dimension mismatch detection works |

## Key Findings

### 1. Performance Benchmarks
- **Small (n=1,000):** All metrics complete <1 second ✅
- **Medium (n=10,000):** All metrics complete <5 seconds ✅
- **Large (n=100,000):** Hit 30-second timeout - needs optimization

### 2. Memory Issue Detected
The memory leak test shows significant memory growth (27,526x) after 1000 snapshot cycles. This indicates potential issues with:
- Circular references in Snapshot objects
- Memory not being freed properly
- Need for explicit cleanup

### 3. HuggingFaceAdapter Bug (FIXED)
**Issue:** `self._normalize` boolean was shadowing the method call, causing `TypeError: 'bool' object is not callable`

**Fix:** Changed line 107 in `huggingface.py` from:
```python
embeddings = self._normalize(embeddings)
```
to:
```python
embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
```

### 4. Missing Dependencies
- `sentence_transformers` not installed in current environment
- Optional dependency that needs to be installed separately

## Files Generated

- `job_metadata.json` - Job configuration and results
- `environment_info.txt` - Python version, packages, system info
- `scale_parametric_tests.log` - Scale test output (parametric tests)
- `scale_other_tests.log` - Scale test output (memory, scaling)
- `realworld_tests.log` - Real-world test output
- `SUMMARY.md` - This summary document

## Recommendations

1. **Performance Optimization:** Investigate optimizing metrics for n=100K+ samples
2. **Memory Leak:** Debug and fix memory leak in snapshot creation/cleanup
3. **Re-run Tests:** After fixes, re-run real-world tests to verify HuggingFace integration
4. **Install Dependencies:** Add `sentence_transformers` for full test coverage

## Commands to Reproduce

```bash
# Run scale tests (excluding 100K test which may timeout)
uv run pytest tests/stress/test_scale.py -m "not slow" --timeout=300 -v

# Run real-world tests
uv run pytest tests/stress/test_real_world.py --timeout=300 -v
```
