"""Classification result dataclass and confidence levels."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConfidenceLevel(Enum):
    """Classification confidence levels based on local NPS."""
    HIGH = "high"      # NPS > 0.90
    MEDIUM = "medium"  # NPS 0.80-0.90
    LOW = "low"        # NPS < 0.80


@dataclass(frozen=True)
class ClassificationResult:
    """Result of drift-aware classification.
    
    Attributes:
        label: Predicted label
        confidence: Confidence level (HIGH, MEDIUM, LOW)
        local_nps: Local neighborhood preservation score
        drift_warning: Warning message if confidence is not HIGH
        nearest_anchor_indices: Indices of k nearest anchor points
        nearest_anchor_distances: Distances to k nearest anchor points
        metadata: Additional metadata
    """
    label: Any
    confidence: ConfidenceLevel
    local_nps: float
    drift_warning: str = ""
    nearest_anchor_indices: tuple = field(default_factory=tuple)
    nearest_anchor_distances: tuple = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute confidence level from local NPS if not provided."""
        # Compute confidence from local_nps
        if self.local_nps > 0.90:
            confidence = ConfidenceLevel.HIGH
        elif self.local_nps > 0.80:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW

        object.__setattr__(self, 'confidence', confidence)

        # Generate drift warning if needed
        if self.confidence != ConfidenceLevel.HIGH and not self.drift_warning:
            warning = f"Drift detected: local NPS = {self.local_nps:.3f}"
            object.__setattr__(self, 'drift_warning', warning)

    @property
    def is_drifted(self) -> bool:
        """Return True if drift is detected (confidence is not HIGH)."""
        return self.confidence != ConfidenceLevel.HIGH
