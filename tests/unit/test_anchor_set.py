"""Tests for AnchorSet dataclass."""

import pytest
from semantic_sentry.probes.anchor_set import AnchorSet


class TestAnchorSet:
    """Test AnchorSet functionality."""
    
    def test_creation_with_text_inputs(self):
        """Test creating anchor set with text inputs."""
        inputs = ["hello", "world", "test"]
        anchor_set = AnchorSet(inputs=inputs)
        
        assert anchor_set.n_samples == 3
        assert anchor_set.modality == "text"
        assert anchor_set.version_hash != ""
    
    def test_creation_with_labels(self):
        """Test creating anchor set with labels."""
        inputs = ["hello", "world", "test"]
        labels = ("greeting", "noun", "noun")
        anchor_set = AnchorSet(inputs=inputs, labels=labels)
        
        assert anchor_set.labels == labels
    
    def test_version_hash_is_deterministic(self):
        """Version hash must be deterministic for same inputs."""
        inputs = ["hello", "world", "test"]
        anchor_set1 = AnchorSet(inputs=inputs)
        anchor_set2 = AnchorSet(inputs=inputs)
        
        assert anchor_set1.version_hash == anchor_set2.version_hash
    
    def test_different_inputs_produce_different_hashes(self):
        """Different inputs must produce different hashes."""
        anchor_set1 = AnchorSet(inputs=["hello", "world"])
        anchor_set2 = AnchorSet(inputs=["foo", "bar"])
        
        assert anchor_set1.version_hash != anchor_set2.version_hash
    
    def test_version_hash_stability_across_instances(self):
        """Version hash must be stable across different instantiations."""
        inputs = ["a", "b", "c", "d", "e"]
        hashes = [AnchorSet(inputs=inputs).version_hash for _ in range(5)]
        
        assert all(h == hashes[0] for h in hashes)
    
    def test_modality_can_be_specified(self):
        """Test that modality can be specified."""
        anchor_set = AnchorSet(inputs=["img1", "img2"], modality="image")
        
        assert anchor_set.modality == "image"
    
    def test_frozen_immutable(self):
        """AnchorSet must be frozen/immutable."""
        anchor_set = AnchorSet(inputs=["test"])
        
        with pytest.raises(AttributeError):
            anchor_set.n_samples = 10
    
    def test_creation_with_numpy_array(self):
        """Test creating anchor set with numpy array inputs."""
        import numpy as np
        
        inputs = np.random.randn(10, 64)
        anchor_set = AnchorSet(inputs=inputs)
        
        assert anchor_set.n_samples == 10
        assert anchor_set.version_hash != ""
