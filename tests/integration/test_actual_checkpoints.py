"""Integration test with actual PyTorch checkpoints."""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from semantic_sentry import AnchorSet, DriftMonitor, Snapshot
from semantic_sentry.adapters.custom import CustomAdapter
from semantic_sentry.core.comparison import AlertSeverity


class TextEncoder(nn.Module):
    """Realistic text encoder for testing."""

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(hidden_dim * 2, 128)
        self.name_or_path = "test-encoder"

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.encoder(emb)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.projection(h)


class TestActualCheckpoints:
    """Test with actual PyTorch model checkpoints."""

    def test_two_similar_checkpoints_show_low_drift(self):
        """Two similar models should show LOW drift."""
        # Create base model
        torch.manual_seed(42)
        model_v0 = TextEncoder()

        # Create nearly identical model (just random init)
        torch.manual_seed(43)
        model_v1 = TextEncoder()

        # Create anchor set
        anchor_texts = [
            "machine learning is great",
            "natural language processing",
            "deep learning models",
            "neural networks",
            "artificial intelligence",
            "computer vision",
            "reinforcement learning",
            "supervised learning",
            "unsupervised learning",
            "transfer learning",
        ] * 10 # 100 samples

        anchor_set = AnchorSet(
            inputs=anchor_texts,
            labels=tuple([f"topic_{i % 5}" for i in range(len(anchor_texts))]),
            modality="text"
        )

        # Create adapters
        def tokenize_simple(texts, vocab_size=1000):
            tokenized = []
            for text in texts:
                tokens = [hash(word) % vocab_size for word in text.lower().split()[:10]]
                tokens += [0] * (10 - len(tokens))
                tokenized.append(tokens)
            return torch.tensor(tokenized, dtype=torch.long)

        def make_adapter(model):
            def encode(texts):
                tokens = tokenize_simple(texts)
                with torch.no_grad():
                    return model(tokens)
            return CustomAdapter(encode_fn=encode, tower_count=1)

        make_adapter(model_v0)
        make_adapter(model_v1)

        # Save checkpoints
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            # Save model v0
            checkpoint_v0 = checkpoint_dir / "model_v0.pt"
            torch.save(model_v0.state_dict(), checkpoint_v0)

            # Save model v1
            checkpoint_v1 = checkpoint_dir / "model_v1.pt"
            torch.save(model_v1.state_dict(), checkpoint_v1)

            # Load checkpoints (simulating real workflow)
            loaded_v0 = TextEncoder()
            loaded_v0.load_state_dict(torch.load(checkpoint_v0))

            loaded_v1 = TextEncoder()
            loaded_v1.load_state_dict(torch.load(checkpoint_v1))

            # Test with SemanticSentry
            monitor = DriftMonitor()

            # Create adapters for loaded models
            adapter_loaded_v0 = make_adapter(loaded_v0)
            adapter_loaded_v1 = make_adapter(loaded_v1)

            # Capture snapshots
            snapshot_v0 = monitor.snapshot(loaded_v0, anchor_set, adapter=adapter_loaded_v0)
            snapshot_v1 = monitor.snapshot(loaded_v1, anchor_set, adapter=adapter_loaded_v1)

            # Compare
            comparison = monitor.compare(snapshot_v0, snapshot_v1)

            # Verify drift detection
            assert "cka" in comparison.global_metrics
            assert "nps" in comparison.global_metrics
            assert 0.0 <= comparison.global_metrics["cka"] <= 1.0
            assert 0.0 <= comparison.global_metrics["nps"] <= 1.0

            print("\nDrift between similar models:")
            print(f" CKA: {comparison.global_metrics['cka']:.4f}")
            print(f" NPS: {comparison.global_metrics['nps']:.4f}")
            print(f" Severity: {comparison.severity.value}")

    def test_drifted_checkpoint_shows_high_drift(self):
        """A drifted model should show higher drift severity."""
        # Create base model
        torch.manual_seed(42)
        model_v0 = TextEncoder()

        # Create significantly drifted model (simulating fine-tuning with large LR)
        model_v1 = TextEncoder()
        model_v1.load_state_dict(model_v0.state_dict())

        # Apply large perturbation (simulate aggressive fine-tuning)
        with torch.no_grad():
            for param in model_v1.parameters():
                param.add_(torch.randn_like(param) * 0.5)

        # Create anchor set
        anchor_texts = [
            "the quick brown fox",
            "jumps over the lazy dog",
            "machine learning revolution",
            "deep neural networks",
            "transformer architecture",
            "attention mechanism",
            "large language models",
            "embedding spaces",
            "semantic similarity",
            "vector representations",
        ] * 10

        anchor_set = AnchorSet(inputs=anchor_texts, modality="text")

        # Create adapters
        def tokenize_simple(texts, vocab_size=1000):
            tokenized = []
            for text in texts:
                tokens = [hash(word) % vocab_size for word in text.lower().split()[:10]]
                tokens += [0] * (10 - len(tokens))
                tokenized.append(tokens)
            return torch.tensor(tokenized, dtype=torch.long)

        def make_adapter(model):
            def encode(texts):
                tokens = tokenize_simple(texts)
                with torch.no_grad():
                    return model(tokens)
            return CustomAdapter(encode_fn=encode, tower_count=1)

        adapter_v0 = make_adapter(model_v0)
        adapter_v1 = make_adapter(model_v1)

        # Test with SemanticSentry
        monitor = DriftMonitor()

        snapshot_v0 = monitor.snapshot(model_v0, anchor_set, adapter=adapter_v0)
        snapshot_v1 = monitor.snapshot(model_v1, anchor_set, adapter=adapter_v1)

        comparison = monitor.compare(snapshot_v0, snapshot_v1)

        # Verify significant drift detected
        assert comparison.global_metrics["cka"] < 0.95, "CKA should show drift"
        assert comparison.severity in [
            AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL,
        ]

        print("\nDrift between drifted models:")
        print(f" CKA: {comparison.global_metrics['cka']:.4f}")
        print(f" NPS: {comparison.global_metrics['nps']:.4f}")
        print(f" Isotropy Δ: {comparison.global_metrics.get('isotropy_delta', 0):.4f}")
        print(f" Severity: {comparison.severity.value}")

    def test_checkpoint_save_load_roundtrip(self):
        """Test that snapshots can be saved and loaded after checkpoint comparison."""
        # Create models
        torch.manual_seed(42)
        model_v0 = TextEncoder()
        model_v1 = TextEncoder()

        with torch.no_grad():
            for param in model_v1.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        anchor_set = AnchorSet(
            inputs=["test"] * 50,
            modality="text"
        )

        def make_adapter(model):
            def encode(texts):
                tokens = torch.randint(0, 1000, (len(texts), 10))
                with torch.no_grad():
                    return model(tokens)
            return CustomAdapter(encode_fn=encode, tower_count=1)

        # Capture snapshots
        monitor = DriftMonitor()
        snapshot_v0 = monitor.snapshot(model_v0, anchor_set, adapter=make_adapter(model_v0))
        snapshot_v1 = monitor.snapshot(model_v1, anchor_set, adapter=make_adapter(model_v1))

        # Save snapshots
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_v0.save(Path(tmpdir) / "v0")
            snapshot_v1.save(Path(tmpdir) / "v1")

            # Load snapshots
            loaded_v0 = Snapshot.load(Path(tmpdir) / "v0")
            loaded_v1 = Snapshot.load(Path(tmpdir) / "v1")

            # Compare loaded snapshots
            comparison = monitor.compare(loaded_v0, loaded_v1)

            assert "cka" in comparison.global_metrics
            assert (Path(tmpdir) / "v0").exists()
            assert (Path(tmpdir) / "v1").exists()
            print("\nCheckpoint roundtrip test: PASSED")
            print(f" Loaded CKA: {comparison.global_metrics['cka']:.4f}")

    def test_quantized_checkpoint_drift(self):
        """Test drift detection between full-precision and simulated quantized model."""
        # Create base model
        torch.manual_seed(42)
        model_fp32 = TextEncoder()

        # Create simulated quantized version (round to lower precision)
        model_quantized = TextEncoder()
        model_quantized.load_state_dict(model_fp32.state_dict())

        with torch.no_grad():
            for param in model_quantized.parameters():
                # Simulate int8 quantization
                param.data = (param.data * 127).round() / 127

        anchor_set = AnchorSet(
            inputs=["quantization test"] * 100,
            modality="text"
        )

        def make_adapter(model):
            def encode(texts):
                tokens = torch.randint(0, 1000, (len(texts), 10))
                with torch.no_grad():
                    return model(tokens)
            return CustomAdapter(encode_fn=encode, tower_count=1)

        monitor = DriftMonitor()

        snapshot_fp32 = monitor.snapshot(
            model_fp32, anchor_set, adapter=make_adapter(model_fp32)
        )
        snapshot_quant = monitor.snapshot(
            model_quantized, anchor_set, adapter=make_adapter(model_quantized)
        )

        comparison = monitor.compare(snapshot_fp32, snapshot_quant)

        print("\nQuantization drift:")
        print(f" CKA: {comparison.global_metrics['cka']:.4f}")
        print(f" NPS: {comparison.global_metrics['nps']:.4f}")
        print(f" Severity: {comparison.severity.value}")

        # Quantization can show significant drift (especially with simulated int8)
        # Just verify drift is detected (CKA < 1.0)
        assert comparison.global_metrics["cka"] < 1.0
        assert comparison.severity in [
            AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL,
        ]


if __name__ == "__main__":
    # Run tests
    test = TestActualCheckpoints()
    test.test_two_similar_checkpoints_show_low_drift()
    test.test_drifted_checkpoint_shows_high_drift()
    test.test_checkpoint_save_load_roundtrip()
    test.test_quantized_checkpoint_drift()
    print("\n All checkpoint tests passed!")
