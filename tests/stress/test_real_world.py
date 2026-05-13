"""Phase 6: Real-world model testing."""

import pytest


@pytest.mark.real_world
@pytest.mark.slow
class TestRealWorld:
    """Real-world model stress tests (downloads models, SLOW)."""

    @pytest.mark.timeout(120)
    def test_real_001_bert_base(self):
        """REAL-001: Load bert-base-uncased from HF Hub and capture snapshot."""
        import numpy as np
        from transformers import AutoModel, AutoTokenizer

        from semantic_sentry.adapters.huggingface import HuggingFaceAdapter
        from semantic_sentry.core.monitor import DriftMonitor
        from semantic_sentry.probes.anchor_set import AnchorSet

        model = AutoModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        anchor_texts = [f"This is test sentence number {i}." for i in range(50)]
        anchor_set = AnchorSet(
            inputs=anchor_texts,
            labels=np.array(["general"] * 50),
            modality="text",
        )
        monitor = DriftMonitor()
        adapter = HuggingFaceAdapter(model, tokenizer)
        snap = monitor.snapshot(model=model, anchor_set=anchor_set, adapter=adapter)
        assert snap.tower_count == 1
        assert snap.embeddings["encoder"].shape == (50, 768)

    @pytest.mark.timeout(180)
    def test_real_003_sentence_transformer(self):
        """REAL-003: SentenceTransformer all-MiniLM-L6-v2."""
        import numpy as np
        from sentence_transformers import SentenceTransformer

        from semantic_sentry.adapters.sentence_transformer import SentenceTransformerAdapter
        from semantic_sentry.core.monitor import DriftMonitor
        from semantic_sentry.probes.anchor_set import AnchorSet

        model = SentenceTransformer("all-MiniLM-L6-v2")
        anchor_texts = [f"Sentence {i} for embedding." for i in range(50)]
        anchor_set = AnchorSet(
            inputs=anchor_texts,
            labels=np.array(["general"] * 50),
            modality="text",
        )
        monitor = DriftMonitor()
        adapter = SentenceTransformerAdapter(model)
        snap = monitor.snapshot(model=model, anchor_set=anchor_set, adapter=adapter)
        assert snap.tower_count == 1
        assert snap.embeddings["encoder"].shape[0] == 50

    @pytest.mark.timeout(300)
    def test_real_012_different_dims_raises(self):
        """REAL-012: Comparing base vs large (different dimensions) must raise EmbeddingDimError."""

        import numpy as np

        from semantic_sentry.adapters.custom import CustomAdapter
        from semantic_sentry.core.monitor import DriftMonitor
        from semantic_sentry.exceptions import EmbeddingDimError
        from semantic_sentry.probes.anchor_set import AnchorSet

        rng = np.random.default_rng(42)
        # "base" model: 768-dim embeddings
        Z_base = rng.standard_normal((50, 768)).astype(np.float32)
        Z_base = Z_base / np.linalg.norm(Z_base, axis=1, keepdims=True)
        # "large" model: 1024-dim embeddings
        Z_large = rng.standard_normal((50, 1024)).astype(np.float32)
        Z_large = Z_large / np.linalg.norm(Z_large, axis=1, keepdims=True)

        anchor_set = AnchorSet(
            inputs=[f"s_{i}" for i in range(50)],
            labels=np.array(["a"] * 50),
            modality="text",
        )

        adapter_base = CustomAdapter(
            encode_fn=lambda inputs: {"encoder": Z_base[:len(inputs)]},
            tower_names=["encoder"],
        )
        adapter_large = CustomAdapter(
            encode_fn=lambda inputs: {"encoder": Z_large[:len(inputs)]},
            tower_names=["encoder"],
        )

        monitor = DriftMonitor()
        snap_base = monitor.snapshot(model=None, anchor_set=anchor_set, adapter=adapter_base)
        snap_large = monitor.snapshot(model=None, anchor_set=anchor_set, adapter=adapter_large)

        with pytest.raises(EmbeddingDimError):
            monitor.compare(snap_base, snap_large)
