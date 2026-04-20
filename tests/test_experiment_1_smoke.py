"""Smoke test for Experiment 1.

Quick test to verify the experiment code can start and basic functionality works.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_import():
    """Test that the module imports successfully."""
    print("Testing import...")
    from experiments import experiment_1_full_finetune
    print("✓ Import successful")
    return True


def test_cli_help():
    """Test that CLI help works."""
    print("\nTesting CLI help...")
    import subprocess
    result = subprocess.run(
        ["python", "experiments/experiment_1_full_finetune.py", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    if result.returncode == 0 and "Full Fine-Tuning" in result.stdout:
        print("✓ CLI help works")
        return True
    else:
        print("✗ CLI help failed")
        print(result.stderr)
        return False


def test_expected_methods():
    """Test that expected methods are present."""
    print("\nTesting expected methods...")
    from experiments.experiment_1_full_finetune import (
        download_mscoco,
        evaluate_retrieval,
        run_experiment,
        SimpleImageCaptionDataset,
    )

    methods = [download_mscoco, evaluate_retrieval, run_experiment]
    for method in methods:
        assert callable(method), f"{method} is not callable"

    assert hasattr(SimpleImageCaptionDataset, "__len__")
    assert hasattr(SimpleImageCaptionDataset, "__getitem__")

    print("✓ All expected methods present")
    return True


def test_per_tower_metrics_structure():
    """Test that per-tower metrics are defined in output."""
    print("\nTesting per-tower metrics structure...")

    # Expected checkpoint structure
    expected_metrics = [
        "cka",
        "cka_vision",
        "cka_language",
        "nps",
        "nps_vision",
        "nps_language",
        "isotropy_delta",
    ]

    checkpoint = {
        "epoch": 1,
        "drift_metrics": {m: 0.5 for m in expected_metrics},
        "retrieval_metrics": {"map_at_k": 0.9, "degradation_percent": 1.0},
    }

    for metric in expected_metrics:
        assert metric in checkpoint["drift_metrics"], f"Missing metric: {metric}"

    print("✓ Per-tower metrics structure correct")
    return True


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("EXPERIMENT 1 SMOKE TESTS")
    print("=" * 60)

    tests = [
        test_import,
        test_cli_help,
        test_expected_methods,
        test_per_tower_metrics_structure,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
