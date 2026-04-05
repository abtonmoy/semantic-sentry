"""Tests for custom exceptions."""

import pytest
from semantic_sentry.exceptions import (
    SemanticSentryError,
    AdapterDetectionError,
    TowerMismatchError,
    EmbeddingDimError,
    AnchorSetMismatchError,
    MetricRegistrationError,
    CalibrationNotFoundError,
    SnapshotCorruptionError,
    NoComparisonError,
)


class TestExceptions:
    """Test that all exceptions are properly defined and catchable."""
    
    @pytest.mark.parametrize("exc_class", [
        AdapterDetectionError,
        TowerMismatchError,
        EmbeddingDimError,
        AnchorSetMismatchError,
        MetricRegistrationError,
        CalibrationNotFoundError,
        SnapshotCorruptionError,
        NoComparisonError,
    ])
    def test_all_exceptions_catchable_via_base(self, exc_class):
        """All exceptions must be catchable via SemanticSentryError."""
        with pytest.raises(SemanticSentryError):
            raise exc_class("test message")
    
    def test_exception_message_preserved(self):
        """Exception messages must be preserved."""
        msg = "custom error message"
        try:
            raise AdapterDetectionError(msg)
        except AdapterDetectionError as e:
            assert str(e) == msg
    
    def test_exception_is_instance_of_exception(self):
        """All exceptions must be instances of Python Exception."""
        exc = TowerMismatchError("test")
        assert isinstance(exc, Exception)
