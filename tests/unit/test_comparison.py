"""Tests for Comparison dataclass."""

import pytest

from semantic_sentry.core.comparison import AlertSeverity, Comparison


class TestAlertSeverity:
    """Test AlertSeverity enum."""

    def test_severity_levels(self):
        """Test that all severity levels exist."""
        assert AlertSeverity.LOW.value == "low"
        assert AlertSeverity.MEDIUM.value == "medium"
        assert AlertSeverity.HIGH.value == "high"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestComparison:
    """Test Comparison functionality."""

    def test_creation_with_low_severity(self):
        """Test creation with metrics indicating low severity."""
        comparison = Comparison(
            snapshot_v0_hash="hash0",
            snapshot_v1_hash="hash1",
            global_metrics={"nps": 0.96, "cka": 0.99},
        )

        assert comparison.severity == AlertSeverity.LOW

    def test_creation_with_medium_severity(self):
        """Test creation with metrics indicating medium severity."""
        comparison = Comparison(
            snapshot_v0_hash="hash0",
            snapshot_v1_hash="hash1",
            global_metrics={"nps": 0.90, "cka": 0.95},  # NPS triggers medium
        )

        assert comparison.severity == AlertSeverity.MEDIUM

    def test_creation_with_high_severity(self):
        """Test creation with metrics indicating high severity."""
        comparison = Comparison(
            snapshot_v0_hash="hash0",
            snapshot_v1_hash="hash1",
            global_metrics={"nps": 0.80, "cka": 0.85},  # NPS triggers high
        )

        assert comparison.severity == AlertSeverity.HIGH

    def test_creation_with_critical_severity(self):
        """Test creation with metrics indicating critical severity."""
        comparison = Comparison(
            snapshot_v0_hash="hash0",
            snapshot_v1_hash="hash1",
            global_metrics={"nps": 0.60, "cka": 0.90},  # NPS triggers critical
        )

        assert comparison.severity == AlertSeverity.CRITICAL

    def test_cka_can_trigger_severity(self):
        """Test that low CKA can trigger severity independently."""
        comparison = Comparison(
            snapshot_v0_hash="hash0",
            snapshot_v1_hash="hash1",
            global_metrics={"nps": 0.99, "cka": 0.70},  # CKA triggers critical
        )

        assert comparison.severity == AlertSeverity.CRITICAL

    def test_custom_thresholds(self):
        """Test that custom thresholds can be used."""
        comparison = Comparison(
            snapshot_v0_hash="hash0",
            snapshot_v1_hash="hash1",
            global_metrics={"nps": 0.50, "cka": 0.99},
            thresholds={"nps_high": 0.40},  # More lenient
        )

        # With default thresholds, 0.50 would be CRITICAL
        # With custom threshold, 0.50 > 0.40, so HIGH
        assert comparison.severity == AlertSeverity.HIGH

    def test_get_metric_global(self):
        """Test getting global metric."""
        comparison = Comparison(
            snapshot_v0_hash="hash0",
            snapshot_v1_hash="hash1",
            global_metrics={"nps": 0.85, "cka": 0.92},
        )

        assert comparison.get_metric("nps") == 0.85
        assert comparison.get_metric("cka") == 0.92

    def test_get_metric_per_tower(self):
        """Test getting per-tower metric."""
        comparison = Comparison(
            snapshot_v0_hash="hash0",
            snapshot_v1_hash="hash1",
            global_metrics={"nps": 0.85},
            per_tower_metrics={
                "vision": {"nps": 0.90},
                "language": {"nps": 0.80},
            },
        )

        assert comparison.get_metric("nps", tower="vision") == 0.90
        assert comparison.get_metric("nps", tower="language") == 0.80

    def test_get_metric_not_found(self):
        """Test that getting non-existent metric raises KeyError."""
        comparison = Comparison(
            snapshot_v0_hash="hash0",
            snapshot_v1_hash="hash1",
            global_metrics={"nps": 0.85},
        )

        with pytest.raises(KeyError):
            comparison.get_metric("nonexistent")

    def test_get_per_tower_metric_no_data(self):
        """Test that requesting per-tower metric when none exists raises error."""
        comparison = Comparison(
            snapshot_v0_hash="hash0",
            snapshot_v1_hash="hash1",
            global_metrics={"nps": 0.85},
            per_tower_metrics=None,
        )

        with pytest.raises(KeyError, match="No per-tower"):
            comparison.get_metric("nps", tower="vision")
