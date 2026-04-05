# Stress Test Results Summary

**Job ID:** stress_test_20260405_040152  
**Date:** 2026-04-05T04:01:52Z  
**Status:** Completed

## Test Suite Overview

This job executed the SemanticSentry stress test suite covering:
- Phase 1.1: CKA metric robustness (11 tests)
- Phase 1.2: NPS metric robustness (10 tests)
- Phase 1.3: Isotropy metric robustness (10 tests)
- Phase 1.4: Classification tests (SKIPPED - feature not implemented)
- Phase 2: Scale tests (SKIPPED - run separately due to large resource requirements)
- Phase 3: Concurrency and thread safety (4 tests)
- Phase 4: Data quality edge cases (10 tests)
- Phase 5: Adversarial scenarios (7 tests)
- Phase 5.2: Transfer function tests (10 tests)
- Phase 6: Real-world model tests (SKIPPED - requires model downloads)

## Results Summary

| Metric | Count |
|--------|-------|
| Total Tests | 62 |
| Passed | 58 |
| Failed | 4 |
| Skipped | 0 |

**Pass Rate:** 93.5%

## Failed Tests

### 1. test_dqa_001_nan_values (test_data_quality.py)
- **Issue:** CKA does not raise ValueError/RuntimeError for NaN inputs
- **Expected:** NaN values should raise an error
- **Actual:** CKA runs and returns NaN
- **Recommendation:** Add input validation to metrics

### 2. test_dqa_002_inf_values (test_data_quality.py)
- **Issue:** CKA does not raise ValueError/RuntimeError for Inf inputs
- **Expected:** Inf values should raise an error
- **Actual:** CKA runs with RuntimeWarning and returns result
- **Recommendation:** Add input validation to metrics

### 3. test_adv_004_gaussian_noise_injection (test_adversarial.py)
- **Issue:** CKA returns 1.0 for noisy data (too lenient)
- **Expected:** CKA < 0.95 to detect drift
- **Actual:** CKA = 1.0 (invariant to the noise pattern)
- **Note:** This may be expected behavior - orthogonal rotation preserves CKA

### 4. test_adv_007_modified_hash (test_adversarial.py)
- **Issue:** Snapshot does not raise SnapshotCorruptionError for modified checkpoint hash
- **Expected:** Integrity check should fail
- **Actual:** Snapshot loads without error
- **Recommendation:** The integrity check only validates embeddings_hash, not checkpoint_hash

## Files Generated

- `job_metadata.json` - Job configuration and results summary
- `environment_info.txt` - Python version, packages, system info
- `test_output.log` - Initial test run output (partial/interrupted)
- `test_run_complete.log` - Complete test run output
- `junit_report.xml` - JUnit XML format test results
- `SUMMARY.md` - This summary document

## Next Steps

1. **Fix input validation:** Add NaN/Inf checks to metric functions
2. **Review adversarial tests:** Confirm expected behavior for noise injection
3. **Fix snapshot integrity:** Extend integrity check to include checkpoint_hash
4. **Run scale tests separately:** Execute test_scale.py with appropriate resources
5. **Run real-world tests:** Execute test_real_world.py when model downloads are acceptable

## Command to Reproduce

```bash
uv run pytest tests/stress/test_cka_stress.py \
  tests/stress/test_nps_stress.py \
  tests/stress/test_isotropy_stress.py \
  tests/stress/test_data_quality.py \
  tests/stress/test_adversarial.py \
  tests/stress/test_concurrency.py \
  tests/stress/test_transfer_stress.py \
  --timeout=300 -v --tb=short
```
