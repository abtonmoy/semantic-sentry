"""Tests for encoder adapters."""

import numpy as np
import torch
import pytest

from semantic_sentry.adapters.base import EncoderAdapter, SingleTowerAdapter, MultiTowerAdapter
from semantic_sentry.adapters.custom import CustomAdapter
from semantic_sentry.adapters import detect_adapter
from semantic_sentry.exceptions import AdapterDetectionError


class MockModel:
    """Mock model for testing."""
    pass


class TestCustomAdapter:
    """Test CustomAdapter functionality."""
    
    def test_single_tower_custom(self):
        """Test single-tower custom adapter."""
        def encode_fn(inputs):
            return torch.randn(len(inputs), 64)
        
        adapter = CustomAdapter(encode_fn=encode_fn, tower_count=1)
        
        assert adapter.tower_count == 1
        assert adapter.list_towers() == ["tower_0"]
    
    def test_multi_tower_custom(self):
        """Test multi-tower custom adapter."""
        def encode_fn(inputs):
            return {
                "vision": torch.randn(len(inputs), 512),
                "text": torch.randn(len(inputs), 512),
            }
        
        adapter = CustomAdapter(
            encode_fn=encode_fn,
            tower_count=2,
            tower_names=["vision", "text"]
        )
        
        assert adapter.tower_count == 2
        assert adapter.list_towers() == ["vision", "text"]
    
    def test_encode_returns_tensors(self):
        """encode returns dict of tensors."""
        def encode_fn(inputs):
            return torch.randn(len(inputs), 64)
        
        adapter = CustomAdapter(encode_fn=encode_fn)
        result = adapter.encode(["test"])
        
        assert "tower_0" in result
        assert isinstance(result["tower_0"], torch.Tensor)
        assert result["tower_0"].shape == (1, 64)
    
    def test_encode_normalizes(self):
        """encode normalizes by default."""
        def encode_fn(inputs):
            return torch.tensor([[3.0, 4.0]])  # Norm = 5
        
        adapter = CustomAdapter(encode_fn=encode_fn, normalize=True)
        result = adapter.encode(["test"])
        
        # Should be L2 normalized
        emb = result["tower_0"]
        norm = torch.norm(emb).item()
        assert abs(norm - 1.0) < 1e-5
    
    def test_encode_no_normalize(self):
        """encode respects normalize=False."""
        def encode_fn(inputs):
            return torch.tensor([[3.0, 4.0]])  # Norm = 5
        
        adapter = CustomAdapter(encode_fn=encode_fn, normalize=False)
        result = adapter.encode(["test"])
        
        emb = result["tower_0"]
        assert emb[0, 0].item() == 3.0
        assert emb[0, 1].item() == 4.0
    
    def test_encode_numpy(self):
        """encode_numpy returns numpy arrays."""
        def encode_fn(inputs):
            return torch.randn(len(inputs), 64)
        
        adapter = CustomAdapter(encode_fn=encode_fn)
        result = adapter.encode_numpy(["test"])
        
        assert "tower_0" in result
        assert isinstance(result["tower_0"], np.ndarray)


class TestBaseAdapter:
    """Test base adapter classes."""
    
    def test_single_tower_list_towers(self):
        """SingleTowerAdapter returns ['encoder']."""
        class TestAdapter(SingleTowerAdapter):
            def encode(self, inputs):
                return {"encoder": torch.randn(1, 64)}
        
        adapter = TestAdapter()
        assert adapter.list_towers() == ["encoder"]
        assert adapter.tower_name == "encoder"
    
    def test_multi_tower_list_towers(self):
        """MultiTowerAdapter returns configured names."""
        class TestAdapter(MultiTowerAdapter):
            def encode(self, inputs):
                return {
                    "tower1": torch.randn(1, 64),
                    "tower2": torch.randn(1, 64),
                }
        
        adapter = TestAdapter(tower_names=["tower1", "tower2"])
        assert adapter.list_towers() == ["tower1", "tower2"]


class TestAutoDetection:
    """Test auto-detection functionality."""
    
    def test_unsupported_model_raises_error(self):
        """Auto-detection raises error for unsupported model."""
        model = MockModel()
        
        with pytest.raises(AdapterDetectionError) as exc_info:
            detect_adapter(model)
        
        assert "MockModel" in str(exc_info.value)
    
    def test_error_message_includes_custom_hint(self):
        """Error message suggests CustomAdapter."""
        model = MockModel()
        
        with pytest.raises(AdapterDetectionError) as exc_info:
            detect_adapter(model)
        
        assert "CustomAdapter" in str(exc_info.value)


class TestAdapterNormalization:
    """Test adapter normalization utilities."""
    
    def test_l2_normalize(self):
        """Test L2 normalization in base adapter."""
        class TestAdapter(EncoderAdapter):
            def encode(self, inputs):
                return {"tower": torch.randn(1, 64)}
            
            def list_towers(self):
                return ["tower"]
        
        adapter = TestAdapter()
        
        # Test normalization
        tensor = torch.tensor([[3.0, 4.0]])  # Norm = 5
        normalized = adapter._normalize(tensor)
        
        expected = torch.tensor([[0.6, 0.8]])  # Unit norm
        assert torch.allclose(normalized, expected, atol=1e-5)
