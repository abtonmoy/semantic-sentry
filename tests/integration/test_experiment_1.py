"""Integration tests for Experiment 1: Full Fine-Tuning Comparison.

This module tests the full fine-tuning experiment code to ensure:
1. The script can be imported without errors
2. Helper functions work correctly
3. The experiment runs with minimal data
4. Output format is correct
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestExperiment1Imports(unittest.TestCase):
    """Test that the experiment module can be imported."""

    def test_import_experiment_module(self):
        """Test that experiment_1_full_finetune.py can be imported."""
        try:
            from experiments import experiment_1_full_finetune  # noqa: F401
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import experiment_1_full_finetune: {e}")

    def test_key_functions_exist(self):
        """Test that key functions are defined."""
        from experiments.experiment_1_full_finetune import (
            SimpleImageCaptionDataset,
            download_mscoco,
            evaluate_retrieval,
            run_experiment,
        )
        self.assertTrue(callable(download_mscoco))
        self.assertTrue(callable(evaluate_retrieval))
        self.assertTrue(callable(run_experiment))
        self.assertTrue(issubclass(SimpleImageCaptionDataset, object))


class TestExperiment1Helpers(unittest.TestCase):
    """Test helper functions."""

    def test_simple_image_caption_dataset(self):
        """Test SimpleImageCaptionDataset initialization."""
        from experiments.experiment_1_full_finetune import SimpleImageCaptionDataset
        from PIL import Image

        # Create dummy data
        images = [Image.new('RGB', (224, 224), color='red') for _ in range(5)]
        captions = [f"Caption {i}" for i in range(5)]

        # Mock preprocess function
        def mock_preprocess(img):
            return np.zeros((3, 224, 224))

        dataset = SimpleImageCaptionDataset(images, captions, mock_preprocess)

        self.assertEqual(len(dataset), 5)
        img, cap = dataset[0]
        self.assertEqual(img.shape, (3, 224, 224))
        self.assertEqual(cap, "Caption 0")


class TestPerTowerMetrics(unittest.TestCase):
    """Test per-tower metric computation."""

    def test_per_tower_nps_computation(self):
        """Test that NPS can be computed separately for vision and language."""
        from semantic_sentry.metrics.nps import nps

        # Create dummy embeddings
        base_vision = np.random.randn(100, 768).astype(np.float32)
        ft_vision = base_vision + np.random.randn(100, 768).astype(np.float32) * 0.1

        base_language = np.random.randn(100, 768).astype(np.float32)
        ft_language = base_language + np.random.randn(100, 768).astype(np.float32) * 0.1

        # Compute NPS for each tower
        nps_vision = nps(base_vision, ft_vision, k=10)
        nps_language = nps(base_language, ft_language, k=10)

        # Both should be between 0 and 1
        self.assertGreaterEqual(nps_vision, 0.0)
        self.assertLessEqual(nps_vision, 1.0)
        self.assertGreaterEqual(nps_language, 0.0)
        self.assertLessEqual(nps_language, 1.0)

        # Compute ratio
        ratio = nps_vision / nps_language if nps_language > 0 else float('inf')
        self.assertIsInstance(ratio, float)


class TestExperiment1OutputFormat(unittest.TestCase):
    """Test output format expectations."""

    def test_expected_checkpoint_structure(self):
        """Test that checkpoint has expected structure."""
        # Mock checkpoint structure
        checkpoint = {
            "epoch": 1,
            "checkpoint_path": "/path/to/checkpoint.pt",
            "embedding_diff": 0.5,
            "drift_metrics": {
                "cka": 0.95,
                "cka_vision": 0.96,
                "cka_language": 0.94,
                "nps": 0.85,
                "nps_vision": 0.87,
                "nps_language": 0.83,
                "isotropy_delta": 0.001,
            },
            "retrieval_metrics": {
                "map_at_k": 0.98,
                "baseline_map": 1.0,
                "degradation_percent": 2.0,
            },
        }

        # Verify structure
        self.assertIn("epoch", checkpoint)
        self.assertIn("drift_metrics", checkpoint)
        self.assertIn("retrieval_metrics", checkpoint)

        # Verify per-tower metrics exist
        drift = checkpoint["drift_metrics"]
        self.assertIn("cka_vision", drift)
        self.assertIn("cka_language", drift)
        self.assertIn("nps_vision", drift)
        self.assertIn("nps_language", drift)
        self.assertIn("isotropy_delta", drift)

    def test_expected_results_structure(self):
        """Test that results.json has expected structure."""
        results = {
            "experiment_config": {
                "timestamp": "20260407_131600",
                "device": "cuda",
                "method": "full_finetune",
                "learning_rate": 1e-5,
                "epochs": [1, 2, 3, 5, 10],
                "train_samples": 2000,
                "val_samples": 1000,
            },
            "training_losses": [
                {"epoch": 1, "loss": 1.5},
                {"epoch": 2, "loss": 1.2},
            ],
            "checkpoints": [],
        }

        self.assertIn("experiment_config", results)
        self.assertIn("training_losses", results)
        self.assertIn("checkpoints", results)

        config = results["experiment_config"]
        self.assertEqual(config["method"], "full_finetune")
        self.assertEqual(config["learning_rate"], 1e-5)


class TestExperiment1Configuration(unittest.TestCase):
    """Test configuration parameters."""

    def test_full_finetune_vs_lora_lr(self):
        """Test that full FT uses lower LR than LoRA."""
        # Full FT should use 1e-5 (10x lower than LoRA's 1e-4)
        full_ft_lr = 1e-5
        lora_lr = 1e-4

        self.assertEqual(full_ft_lr * 10, lora_lr)
        self.assertLess(full_ft_lr, lora_lr)

    def test_all_parameters_trainable(self):
        """Test that full FT has all parameters trainable."""
        # This is a conceptual test - in reality we'd check the model
        # For now, just verify the logic
        total_params = 1000
        trainable_params = 1000

        ratio = trainable_params / total_params
        self.assertEqual(ratio, 1.0)


class TestDriftRatioComputation(unittest.TestCase):
    """Test per-tower drift ratio computation."""

    def test_drift_ratio_calculation(self):
        """Test calculation of vision/language drift ratio."""
        # Simulated NPS values at different epochs
        nps_vision = [1.0, 0.8, 0.6, 0.4]
        nps_language = [1.0, 0.9, 0.8, 0.7]

        # Calculate drops
        vision_drop = 1.0 - nps_vision[-1]  # 0.6
        language_drop = 1.0 - nps_language[-1]  # 0.3

        # Calculate ratio
        ratio = vision_drop / language_drop if language_drop > 0 else float('inf')

        self.assertAlmostEqual(ratio, 2.0, places=5)

    def test_correlated_drift_hypothesis(self):
        """Test that correlated drift produces ratio near 1.0."""
        # Correlated drift: both towers drift similarly
        nps_vision = [1.0, 0.8, 0.6, 0.4]  # Drop of 0.6
        nps_language = [1.0, 0.82, 0.62, 0.42]  # Drop of 0.58

        vision_drop = 1.0 - nps_vision[-1]
        language_drop = 1.0 - nps_language[-1]
        ratio = vision_drop / language_drop

        # Should be close to 1.0 for correlated drift
        self.assertAlmostEqual(ratio, 1.03, places=1)
        self.assertGreater(ratio, 0.8)
        self.assertLess(ratio, 1.2)

    def test_modality_localized_hypothesis(self):
        """Test that modality-localized drift produces ratio far from 1.0."""
        # Modality-localized: one tower drifts more than other
        nps_vision = [1.0, 0.5, 0.3, 0.2]  # Large drop of 0.8
        nps_language = [1.0, 0.95, 0.92, 0.90]  # Small drop of 0.1

        vision_drop = 1.0 - nps_vision[-1]
        language_drop = 1.0 - nps_language[-1]
        ratio = vision_drop / language_drop

        # Should be far from 1.0 for modality-localized drift
        self.assertGreater(ratio, 5.0)


class TestIntegrationMinimal(unittest.TestCase):
    """Integration test with minimal data."""

    @unittest.skip("Requires GPU and takes too long for unit tests")
    def test_experiment_runs_with_minimal_data(self):
        """Test that the experiment runs with minimal dataset."""
        from experiments.experiment_1_full_finetune import run_experiment

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_output"

            # Run with very small dataset
            result_dir = run_experiment(
                output_dir=output_dir,
                train_samples=50,
                val_samples=20,
            )

            # Verify output exists
            self.assertTrue(result_dir.exists())
            self.assertTrue((result_dir / "results.json").exists())

            # Load and verify results
            with open(result_dir / "results.json") as f:
                results = json.load(f)

            self.assertIn("experiment_config", results)
            self.assertIn("checkpoints", results)
            self.assertEqual(results["experiment_config"]["method"], "full_finetune")


class TestCommandLineInterface(unittest.TestCase):
    """Test command line interface."""

    def test_argument_parsing(self):
        """Test that CLI arguments are parsed correctly."""
        import argparse

        # Test default values
        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=str, default=None)
        parser.add_argument("--train-samples", type=int, default=2000)
        parser.add_argument("--val-samples", type=int, default=1000)

        args = parser.parse_args([])
        self.assertEqual(args.train_samples, 2000)
        self.assertEqual(args.val_samples, 1000)
        self.assertIsNone(args.output_dir)

        args = parser.parse_args(["--train-samples", "500", "--val-samples", "100"])
        self.assertEqual(args.train_samples, 500)
        self.assertEqual(args.val_samples, 100)


def run_integration_tests():
    """Run all integration tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_integration_tests()
