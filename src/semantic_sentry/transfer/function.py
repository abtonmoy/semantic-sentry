"""Transfer function for predicting downstream task degradation."""

from abc import ABC, abstractmethod

import numpy as np

from semantic_sentry.core.comparison import Comparison


class TransferFunction(ABC):
    """Abstract base class for transfer functions.
    
    Transfer functions map drift metrics to predicted downstream task
    degradation. This allows drift detection to be actionable without
    requiring labeled evaluation data.
    """

    @abstractmethod
    def fit(self, comparisons: list[Comparison], degradations: list[float]) -> None:
        """Fit the transfer function to calibration data.
        
        Args:
            comparisons: List of comparison results
            degradations: List of measured downstream degradations
        """
        pass

    @abstractmethod
    def predict(self, comparison: Comparison) -> float:
        """Predict downstream degradation for a comparison.
        
        Args:
            comparison: Comparison result to predict from
            
        Returns:
            Predicted degradation (0.0 = no degradation, 1.0 = complete failure)
        """
        pass


class LinearTransfer(TransferFunction):
    """Linear transfer function using OLS regression.
    
    Features: [1-CKA, 1-NPS, |isotropy_delta|]
    Target: Downstream task degradation
    
    Example:
        transfer = LinearTransfer()
        transfer.fit(calibration_comparisons, calibration_degradations)
        predicted_degradation = transfer.predict(comparison)
    """

    def __init__(self):
        """Initialize linear transfer function."""
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self._fitted = False
        self.r_squared: float | None = None

    def fit(self, comparisons: list[Comparison], degradations: list[float]) -> None:
        """Fit linear transfer function.
        
        Args:
            comparisons: List of comparison results
            degradations: List of measured degradations
            
        Raises:
            ValueError: If input lengths don't match or too few samples
        """
        if len(comparisons) != len(degradations):
            raise ValueError(
                f"Length mismatch: {len(comparisons)} comparisons vs {len(degradations)} degradations"
            )

        if len(comparisons) < 3:
            raise ValueError("Need at least 3 samples to fit transfer function")

        # Extract features from comparisons
        X = np.array([self._extract_features(c) for c in comparisons])
        y = np.array(degradations)

        # Fit OLS: y = X @ w + b
        # Add bias term
        X_aug = np.column_stack([X, np.ones(len(X))])

        # Solve using least squares
        solution, residuals, rank, s = np.linalg.lstsq(X_aug, y, rcond=None)

        self.weights = solution[:3]
        self.bias = solution[3]
        self._fitted = True

        # Compute R-squared
        y_pred = X_aug @ solution
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot > 0:
            self.r_squared = 1 - (ss_res / ss_tot)
        else:
            self.r_squared = 0.0

    def predict(self, comparison: Comparison) -> float:
        """Predict downstream degradation.
        
        Args:
            comparison: Comparison result
            
        Returns:
            Predicted degradation
            
        Raises:
            ValueError: If not fitted
        """
        if not self._fitted:
            raise ValueError("TransferFunction not fitted. Call fit() first.")

        features = self._extract_features(comparison)
        prediction = self.weights @ features + self.bias

        # Clamp to [0, 1]
        return float(np.clip(prediction, 0.0, 1.0))

    def _extract_features(self, comparison: Comparison) -> np.ndarray:
        """Extract features from comparison.
        
        Features:
            - 1 - CKA (global structural change)
            - 1 - NPS (local neighborhood change)
            - |isotropy_delta| (spectral geometry change)
        
        Args:
            comparison: Comparison result
            
        Returns:
            Feature vector
        """
        cka = comparison.global_metrics.get("cka", 1.0)
        nps = comparison.global_metrics.get("nps", 1.0)
        iso_delta = comparison.global_metrics.get("isotropy_delta", 0.0)

        return np.array([
            1.0 - cka,
            1.0 - nps,
            abs(iso_delta)
        ])

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance (absolute weights).
        
        Returns:
            Dict mapping feature name to importance
        
        Raises:
            ValueError: If not fitted
        """
        if not self._fitted:
            raise ValueError("Not fitted")

        feature_names = ["1-cka", "1-nps", "|isotropy_delta|"]
        return {
            name: float(abs(weight))
            for name, weight in zip(feature_names, self.weights)
        }


class LogisticTransfer(TransferFunction):
    """Logistic transfer function for binary degradation prediction.
    
    Predicts probability of significant degradation (> threshold).
    """

    def __init__(self, degradation_threshold: float = 0.1):
        """Initialize logistic transfer function.
        
        Args:
            degradation_threshold: Threshold for significant degradation
        """
        self.degradation_threshold = degradation_threshold
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self._fitted = False

    def fit(self, comparisons: list[Comparison], degradations: list[float]) -> None:
        """Fit logistic transfer function.
        
        Args:
            comparisons: List of comparison results
            degradations: List of measured degradations
        """
        if len(comparisons) != len(degradations):
            raise ValueError("Length mismatch")

        if len(comparisons) < 3:
            raise ValueError("Need at least 3 samples")

        # Extract features
        X = np.array([
            [
                1.0 - c.global_metrics.get("cka", 1.0),
                1.0 - c.global_metrics.get("nps", 1.0),
                abs(c.global_metrics.get("isotropy_delta", 0.0))
            ]
            for c in comparisons
        ])

        # Binary targets
        y = np.array([1.0 if d > self.degradation_threshold else 0.0 for d in degradations])

        # Fit using least squares approximation (simplified)
        X_aug = np.column_stack([X, np.ones(len(X))])
        solution, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)

        self.weights = solution[:3]
        self.bias = solution[3]
        self._fitted = True

    def predict(self, comparison: Comparison) -> float:
        """Predict probability of significant degradation.
        
        Args:
            comparison: Comparison result
            
        Returns:
            Probability of degradation
        """
        if not self._fitted:
            raise ValueError("Not fitted")

        cka = comparison.global_metrics.get("cka", 1.0)
        nps = comparison.global_metrics.get("nps", 1.0)
        iso_delta = comparison.global_metrics.get("isotropy_delta", 0.0)

        features = np.array([1.0 - cka, 1.0 - nps, abs(iso_delta)])

        # Linear prediction
        linear_pred = self.weights @ features + self.bias

        # Apply sigmoid
        prob = 1.0 / (1.0 + np.exp(-linear_pred))

        return float(prob)


def create_transfer_function(method: str = "linear", **kwargs) -> TransferFunction:
    """Factory function to create transfer functions.
    
    Args:
        method: Transfer function type ('linear' or 'logistic')
        **kwargs: Additional arguments for the transfer function
        
    Returns:
        TransferFunction instance
        
    Raises:
        ValueError: If method is unknown
    """
    if method == "linear":
        return LinearTransfer()
    elif method == "logistic":
        return LogisticTransfer(**kwargs)
    else:
        raise ValueError(f"Unknown transfer function: {method}")
