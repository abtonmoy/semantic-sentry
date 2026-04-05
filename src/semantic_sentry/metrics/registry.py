"""Metric Registry singleton for managing drift metrics."""

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from semantic_sentry.exceptions import MetricRegistrationError
from semantic_sentry.metrics.cka import linear_cka
from semantic_sentry.metrics.isotropy import isotropy_delta
from semantic_sentry.metrics.nps import nps


@dataclass
class MetricEntry:
    """Entry in the metric registry."""
    fn: Callable[[np.ndarray, np.ndarray], float]
    range: tuple[float, float] | None
    description: str | None


class MetricRegistry:
    """Thread-safe singleton registry for drift metrics.
    
    The registry maintains a collection of metrics that can be computed
    between pairs of embedding matrices. Built-in metrics are registered
    automatically, and custom metrics can be added with validation.
    """

    _instance: "MetricRegistry | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "MetricRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the registry with empty storage."""
        self._metrics: dict[str, MetricEntry] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in metrics (called during initialization, no lock needed)."""
        # Use internal register to avoid lock re-entrance deadlock
        self._internal_register(
            "cka",
            linear_cka,
            range=(0.0, 1.0),
            description="Global structural similarity (Linear CKA)"
        )
        self._internal_register(
            "nps",
            nps,
            range=(0.0, 1.0),
            description="Local neighborhood preservation score"
        )
        self._internal_register(
            "isotropy_delta",
            isotropy_delta,
            range=(-1.0, 1.0),
            description="Spectral geometry change (isotropy difference)"
        )

    def _internal_register(
        self,
        name: str,
        fn: Callable[[np.ndarray, np.ndarray], float],
        range: tuple[float, float] | None = None,
        description: str | None = None
    ) -> None:
        """Internal registration without lock (for builtins during init)."""
        # Builtins are assumed valid, skip determinism check
        self._metrics[name] = MetricEntry(fn=fn, range=range, description=description)

    def register(
        self,
        name: str,
        fn: Callable[[np.ndarray, np.ndarray], float],
        range: tuple[float, float] | None = None,
        description: str | None = None
    ) -> None:
        """Register a new metric.

        Args:
            name: Unique name for the metric
            fn: Function that takes (Z0, Z1) and returns float
            range: Optional (min, max) tuple for validation
            description: Optional description of the metric

        Raises:
            MetricRegistrationError: If validation fails
        """
        # Validate determinism
        self._validate_determinism(fn)

        # Register the metric
        with self._lock:
            self._metrics[name] = MetricEntry(fn=fn, range=range, description=description)

    def _validate_determinism(self, fn: Callable[[np.ndarray, np.ndarray], float]) -> None:
        """Validate that a metric function is deterministic.
        
        Args:
            fn: Function to validate
            
        Raises:
            MetricRegistrationError: If function is not deterministic
        """
        np.random.seed(42)
        Z0 = np.random.randn(50, 32).astype(np.float32)
        Z1 = np.random.randn(50, 32).astype(np.float32)

        # Call twice with same input
        try:
            result1 = fn(Z0, Z1)
            result2 = fn(Z0, Z1)
        except Exception as e:
            raise MetricRegistrationError(f"Metric function raised exception: {e}")

        # Check results are identical
        if result1 != result2:
            raise MetricRegistrationError(
                f"Metric function is not deterministic: {result1} != {result2}"
            )

        # Check return type is float-like
        if not isinstance(result1, (float, np.floating)):
            raise MetricRegistrationError(
                f"Metric function must return float, got {type(result1)}"
            )

    def compute(self, name: str, Z0: np.ndarray, Z1: np.ndarray) -> float:
        """Compute a single metric.
        
        Args:
            name: Name of the metric
            Z0: First embedding matrix
            Z1: Second embedding matrix
            
        Returns:
            Metric value
            
        Raises:
            KeyError: If metric not found
        """
        with self._lock:
            entry = self._metrics[name]

        result = entry.fn(Z0, Z1)

        # Validate range if specified
        if entry.range is not None:
            min_val, max_val = entry.range
            if not (min_val <= result <= max_val):
                raise ValueError(
                    f"Metric '{name}' returned {result}, outside range [{min_val}, {max_val}]"
                )

        return float(result)

    def compute_all(
        self,
        Z0: np.ndarray,
        Z1: np.ndarray,
        metric_names: list[str] | None = None,
        parallel: bool = True
    ) -> dict[str, float]:
        """Compute all registered metrics in parallel.
        
        Args:
            Z0: First embedding matrix
            Z1: Second embedding matrix
            metric_names: Optional list of specific metrics to compute
            parallel: Whether to use parallel execution
            
        Returns:
            Dict mapping metric name to value
        """
        with self._lock:
            names = metric_names or list(self._metrics.keys())
            entries = {name: self._metrics[name] for name in names}

        if not parallel or len(names) == 1:
            # Sequential computation
            return {name: entries[name].fn(Z0, Z1) for name in names}

        # Parallel computation
        def compute_one(name: str) -> tuple[str, float]:
            return name, entries[name].fn(Z0, Z1)

        with ThreadPoolExecutor(max_workers=min(len(names), 4)) as executor:
            results = list(executor.map(compute_one, names))

        return dict(results)

    def list_metrics(self) -> list[str]:
        """List all registered metric names.
        
        Returns:
            List of metric names
        """
        with self._lock:
            return list(self._metrics.keys())

    def get_info(self, name: str) -> MetricEntry:
        """Get information about a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Metric entry
            
        Raises:
            KeyError: If metric not found
        """
        with self._lock:
            return self._metrics[name]

    def unregister(self, name: str) -> None:
        """Unregister a metric.
        
        Args:
            name: Metric name to unregister
            
        Raises:
            KeyError: If metric not found
        """
        with self._lock:
            del self._metrics[name]


def get_metric_registry() -> MetricRegistry:
    """Get the global metric registry instance.
    
    Returns:
        MetricRegistry singleton
    """
    return MetricRegistry()
