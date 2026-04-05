"""Tests for Snapshot dataclass."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from semantic_sentry.core.snapshot import Snapshot
from semantic_sentry.exceptions import SnapshotCorruptionError


class TestSnapshot:
    """Test Snapshot functionality."""
    
    def test_creation_single_tower(self):
        """Test creating a single-tower snapshot."""
        embeddings = {
            "encoder": np.random.randn(100, 64).astype(np.float32)
        }
        snapshot = Snapshot(
            model_id="test-model",
            checkpoint_hash="abc123",
            tower_count=1,
            tower_names=("encoder",),
            embeddings=embeddings,
        )
        
        assert snapshot.model_id == "test-model"
        assert snapshot.tower_count == 1
        assert not snapshot.is_multi_tower
        assert snapshot.cross_tower_alignment is None
    
    def test_creation_multi_tower(self):
        """Test creating a multi-tower snapshot."""
        embeddings = {
            "vision": np.random.randn(100, 512).astype(np.float32),
            "language": np.random.randn(100, 512).astype(np.float32),
        }
        alignment = {("vision", "language"): 0.75}
        snapshot = Snapshot(
            model_id="clip-model",
            checkpoint_hash="def456",
            tower_count=2,
            tower_names=("vision", "language"),
            embeddings=embeddings,
            cross_tower_alignment=alignment,
        )
        
        assert snapshot.tower_count == 2
        assert snapshot.is_multi_tower
        assert snapshot.cross_tower_alignment == alignment
    
    def test_rejects_mismatched_tower_count(self):
        """Snapshot must reject mismatched tower_count vs tower_names."""
        embeddings = {"encoder": np.random.randn(10, 64).astype(np.float32)}
        
        with pytest.raises(ValueError, match="tower_count"):
            Snapshot(
                model_id="test",
                checkpoint_hash="abc123",
                tower_count=2,  # Wrong!
                tower_names=("encoder",),
                embeddings=embeddings,
            )
    
    def test_rejects_mismatched_embeddings_keys(self):
        """Snapshot must reject mismatched embeddings keys vs tower_names."""
        embeddings = {"wrong_name": np.random.randn(10, 64).astype(np.float32)}
        
        with pytest.raises(ValueError, match="Embeddings keys"):
            Snapshot(
                model_id="test",
                checkpoint_hash="abc123",
                tower_count=1,
                tower_names=("encoder",),
                embeddings=embeddings,
            )
    
    def test_rejects_mismatched_n_dimensions(self):
        """Snapshot must reject embeddings with different n dimensions."""
        embeddings = {
            "tower1": np.random.randn(100, 64).astype(np.float32),
            "tower2": np.random.randn(50, 64).astype(np.float32),  # Different n!
        }
        
        with pytest.raises(ValueError, match="same n dimension"):
            Snapshot(
                model_id="test",
                checkpoint_hash="abc123",
                tower_count=2,
                tower_names=("tower1", "tower2"),
                embeddings=embeddings,
            )
    
    def test_get_tower(self):
        """Test getting tower embeddings by name."""
        emb = np.random.randn(100, 64).astype(np.float32)
        embeddings = {"encoder": emb}
        snapshot = Snapshot(
            model_id="test",
            checkpoint_hash="abc123",
            tower_count=1,
            tower_names=("encoder",),
            embeddings=embeddings,
        )
        
        retrieved = snapshot.get_tower("encoder")
        assert np.array_equal(retrieved, emb)
    
    def test_get_tower_not_found(self):
        """Test that getting non-existent tower raises KeyError."""
        embeddings = {"encoder": np.random.randn(10, 64).astype(np.float32)}
        snapshot = Snapshot(
            model_id="test",
            checkpoint_hash="abc123",
            tower_count=1,
            tower_names=("encoder",),
            embeddings=embeddings,
        )
        
        with pytest.raises(KeyError, match="not found"):
            snapshot.get_tower("nonexistent")
    
    def test_save_and_load_roundtrip(self):
        """Test save/load roundtrip preserves all data."""
        embeddings = {
            "encoder": np.random.randn(100, 64).astype(np.float32),
        }
        snapshot = Snapshot(
            model_id="test-model",
            checkpoint_hash="abc123xyz789",
            anchor_set_version="anchor_v1",
            tower_count=1,
            tower_names=("encoder",),
            embeddings=embeddings,
            metadata={"test": "value"},
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "snapshot"
            snapshot.save(save_path)
            
            # Load and verify
            loaded = Snapshot.load(save_path)
            
            assert loaded.model_id == snapshot.model_id
            assert loaded.checkpoint_hash == snapshot.checkpoint_hash
            assert loaded.anchor_set_version == snapshot.anchor_set_version
            assert loaded.tower_count == snapshot.tower_count
            assert loaded.tower_names == snapshot.tower_names
            assert loaded.metadata == snapshot.metadata
            assert np.array_equal(loaded.get_tower("encoder"), snapshot.get_tower("encoder"))
    
    def test_load_detects_corruption(self):
        """Test that loading detects corrupted snapshots."""
        embeddings = {
            "encoder": np.random.randn(100, 64).astype(np.float32),
        }
        snapshot = Snapshot(
            model_id="test",
            checkpoint_hash="abc123xyz789",
            tower_count=1,
            tower_names=("encoder",),
            embeddings=embeddings,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "snapshot"
            snapshot.save(save_path)
            
            # Corrupt the metadata file
            import json
            metadata_path = save_path / "metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Change the embeddings_hash to simulate corruption
            metadata["embeddings_hash"] = "tampered_hash"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)
            
            # Loading should raise SnapshotCorruptionError
            with pytest.raises(SnapshotCorruptionError):
                Snapshot.load(save_path)
