"""Tests for ClassificationResult dataclass."""

import pytest
from semantic_sentry.core.classification import ClassificationResult, ConfidenceLevel


class TestConfidenceLevel:
    """Test ConfidenceLevel enum."""
    
    def test_confidence_levels(self):
        """Test that all confidence levels exist."""
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.LOW.value == "low"


class TestClassificationResult:
    """Test ClassificationResult functionality."""
    
    def test_creation_with_high_confidence(self):
        """Test creation with high local NPS."""
        result = ClassificationResult(
            label="positive",
            confidence=ConfidenceLevel.HIGH,
            local_nps=0.95,
        )
        
        assert result.label == "positive"
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.local_nps == 0.95
    
    def test_confidence_computed_from_nps_high(self):
        """Test that confidence is computed from local_nps when HIGH."""
        result = ClassificationResult(
            label="test",
            confidence=ConfidenceLevel.HIGH,
            local_nps=0.95,  # > 0.90 should be HIGH
        )
        
        assert result.confidence == ConfidenceLevel.HIGH
    
    def test_confidence_computed_from_nps_medium(self):
        """Test that confidence is computed from local_nps when MEDIUM."""
        result = ClassificationResult(
            label="test",
            confidence=ConfidenceLevel.MEDIUM,
            local_nps=0.85,  # 0.80-0.90 should be MEDIUM
        )
        
        assert result.confidence == ConfidenceLevel.MEDIUM
    
    def test_confidence_computed_from_nps_low(self):
        """Test that confidence is computed from local_nps when LOW."""
        result = ClassificationResult(
            label="test",
            confidence=ConfidenceLevel.LOW,
            local_nps=0.70,  # < 0.80 should be LOW
        )
        
        assert result.confidence == ConfidenceLevel.LOW
    
    def test_drift_warning_generated_for_medium(self):
        """Test that drift warning is generated for medium confidence."""
        result = ClassificationResult(
            label="test",
            confidence=ConfidenceLevel.MEDIUM,
            local_nps=0.85,
        )
        
        assert result.drift_warning != ""
        assert "Drift detected" in result.drift_warning
        assert "0.850" in result.drift_warning
    
    def test_drift_warning_generated_for_low(self):
        """Test that drift warning is generated for low confidence."""
        result = ClassificationResult(
            label="test",
            confidence=ConfidenceLevel.LOW,
            local_nps=0.70,
        )
        
        assert result.drift_warning != ""
        assert "Drift detected" in result.drift_warning
    
    def test_is_drifted_true_for_medium(self):
        """Test is_drifted property for medium confidence."""
        result = ClassificationResult(
            label="test",
            confidence=ConfidenceLevel.MEDIUM,
            local_nps=0.85,
        )
        
        assert result.is_drifted is True
    
    def test_is_drifted_true_for_low(self):
        """Test is_drifted property for low confidence."""
        result = ClassificationResult(
            label="test",
            confidence=ConfidenceLevel.LOW,
            local_nps=0.70,
        )
        
        assert result.is_drifted is True
    
    def test_is_drifted_false_for_high(self):
        """Test is_drifted property for high confidence."""
        result = ClassificationResult(
            label="test",
            confidence=ConfidenceLevel.HIGH,
            local_nps=0.95,
        )
        
        assert result.is_drifted is False
    
    def test_nearest_anchors_stored(self):
        """Test that nearest anchor indices and distances are stored."""
        result = ClassificationResult(
            label="test",
            confidence=ConfidenceLevel.HIGH,
            local_nps=0.95,
            nearest_anchor_indices=(1, 5, 10),
            nearest_anchor_distances=(0.1, 0.2, 0.3),
        )
        
        assert result.nearest_anchor_indices == (1, 5, 10)
        assert result.nearest_anchor_distances == (0.1, 0.2, 0.3)
    
    def test_metadata_stored(self):
        """Test that metadata is stored."""
        result = ClassificationResult(
            label="test",
            confidence=ConfidenceLevel.HIGH,
            local_nps=0.95,
            metadata={"extra": "info", "score": 0.99},
        )
        
        assert result.metadata == {"extra": "info", "score": 0.99}
