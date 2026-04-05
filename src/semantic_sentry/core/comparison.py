"""Comparison dataclass for drift detection results."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AlertSeverity(Enum):
    """Alert severity levels based on drift metrics."""
    LOW = "low"          # NPS > 0.95, CKA > 0.98
    MEDIUM = "medium"    # NPS 0.85-0.95 or CKA 0.90-0.98
    HIGH = "high"        # NPS 0.70-0.85 or CKA 0.80-0.90
    CRITICAL = "critical"  # NPS < 0.70 or CKA < 0.80


@dataclass(frozen=True)
class Comparison:
    """Frozen comparison result between two snapshots.
    
    Attributes:
        snapshot_v0_hash: Hash of base snapshot
        snapshot_v1_hash: Hash of updated snapshot
        global_metrics: Global drift metrics (CKA, NPS, isotropy_delta)
        per_tower_metrics: Per-tower metrics for multi-tower models
        alignment_deltas: Changes in cross-tower alignment
        severity: Computed alert severity
        thresholds: Custom thresholds used for severity computation
        metadata: Additional metadata
    """
    snapshot_v0_hash: str
    snapshot_v1_hash: str
    global_metrics: dict[str, float] = field(default_factory=dict)
    per_tower_metrics: dict[str, dict[str, float]] | None = None
    alignment_deltas: dict[tuple[str, str], float] | None = None
    severity: AlertSeverity = AlertSeverity.LOW
    thresholds: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute severity from global metrics."""
        # Use default thresholds if not provided
        default_thresholds = {
            "nps_low": 0.95,
            "nps_medium": 0.85,
            "nps_high": 0.70,
            "cka_low": 0.98,
            "cka_medium": 0.90,
            "cka_high": 0.80,
        }
        thresholds = {**default_thresholds, **self.thresholds}

        # Extract metrics with defaults
        nps = self.global_metrics.get("nps", 1.0)
        cka = self.global_metrics.get("cka", 1.0)

        # Compute severity
        severity = AlertSeverity.LOW
        if nps < thresholds["nps_high"] or cka < thresholds["cka_high"]:
            severity = AlertSeverity.CRITICAL
        elif nps < thresholds["nps_medium"] or cka < thresholds["cka_medium"]:
            severity = AlertSeverity.HIGH
        elif nps < thresholds["nps_low"] or cka < thresholds["cka_low"]:
            severity = AlertSeverity.MEDIUM

        object.__setattr__(self, 'severity', severity)

    def get_metric(self, name: str, tower: str | None = None) -> float:
        """Get a metric value.
        
        Args:
            name: Metric name (e.g., 'nps', 'cka')
            tower: Optional tower name for per-tower metrics
            
        Returns:
            Metric value
            
        Raises:
            KeyError: If metric not found
        """
        if tower is not None:
            if self.per_tower_metrics is None:
                raise KeyError("No per-tower metrics available")
            if tower not in self.per_tower_metrics:
                raise KeyError(f"Tower '{tower}' not found in per-tower metrics")
            return self.per_tower_metrics[tower][name]
        return self.global_metrics[name]
