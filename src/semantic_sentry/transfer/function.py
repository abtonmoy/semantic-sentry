"""Transfer function for predicting downstream task degradation."""

import warnings
from abc import ABC, abstractmethod

import numpy as np

from semantic_sentry.core.comparison import Comparison


# Feature order used by every concrete TransferFunction in this module.
# Kept as a module-level constant so the wider repo can ship a single
# feature_names list with calibration profiles.
TRANSFER_FEATURE_NAMES = ("1-cka", "1-nps", "|isotropy_delta|")


def extract_drift_features(comparison: Comparison) -> np.ndarray:
    """Vectorise a Comparison into the standard 3-feature drift vector.

    Features (in the order declared by `TRANSFER_FEATURE_NAMES`):
        - 1 - CKA (global structural change)
        - 1 - NPS (local neighbourhood change)
        - |isotropy_delta| (spectral geometry change)

    Defaults of 1.0 / 1.0 / 0.0 (= no drift) are used when a metric is
    missing from `comparison.global_metrics`.
    """
    cka = comparison.global_metrics.get("cka", 1.0)
    nps = comparison.global_metrics.get("nps", 1.0)
    iso_delta = comparison.global_metrics.get("isotropy_delta", 0.0)
    return np.array([1.0 - cka, 1.0 - nps, abs(iso_delta)])


class TransferFunction(ABC):
    """Abstract base class for transfer functions.

    Transfer functions map drift metrics to predicted downstream task
    degradation. This allows drift detection to be actionable without
    requiring labelled evaluation data.
    """

    @abstractmethod
    def fit(self, comparisons: list[Comparison], degradations: list[float]) -> None:
        """Fit the transfer function to calibration data."""

    @abstractmethod
    def predict(self, comparison: Comparison) -> float:
        """Predict downstream degradation for a comparison."""

    # Concrete classes share this helper rather than duplicating the
    # feature-vector code (was duplicated in LinearTransfer and
    # LogisticTransfer prior to this refactor).
    def _extract_features(self, comparison: Comparison) -> np.ndarray:
        return extract_drift_features(comparison)


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

        # Fit OLS: y = X @ w + b, with an explicit bias column.
        n_features = X.shape[1]
        X_aug = np.column_stack([X, np.ones(len(X))])
        solution, _residuals, _rank, _s = np.linalg.lstsq(X_aug, y, rcond=None)

        self.weights = solution[:n_features]
        self.bias = float(solution[n_features])
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

    # _extract_features is inherited from TransferFunction.

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance (absolute weights).

        Returns:
            Dict mapping feature name to importance

        Raises:
            ValueError: If not fitted
        """
        if not self._fitted:
            raise ValueError("Not fitted")

        return {
            name: float(abs(weight))
            for name, weight in zip(TRANSFER_FEATURE_NAMES, self.weights)
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

        X = np.array([self._extract_features(c) for c in comparisons])

        # Binary targets: 1 if measured degradation exceeded the threshold.
        y = np.array([1.0 if d > self.degradation_threshold else 0.0
                      for d in degradations])

        # Maximum-likelihood logistic regression via scipy.optimize. Prior
        # to v0.2.0 this method ran OLS on binary labels and applied a
        # sigmoid only at predict() — that was OLS-on-binary, not actual
        # logistic regression. Switching to true MLE.
        from scipy.optimize import minimize

        n_features = X.shape[1]
        # Parameter vector: [w_1, ..., w_d, bias]
        def _neg_log_likelihood(params: np.ndarray) -> float:
            w = params[:n_features]
            b = params[n_features]
            z = X @ w + b
            # log(1 + exp(z)) computed stably via logaddexp(0, z).
            log_one_plus_exp = np.logaddexp(0.0, z)
            nll = -np.sum(y * z - log_one_plus_exp)
            return float(nll)

        def _grad(params: np.ndarray) -> np.ndarray:
            w = params[:n_features]
            b = params[n_features]
            z = X @ w + b
            p = 1.0 / (1.0 + np.exp(-z))
            err = p - y                                  # shape (n,)
            grad_w = X.T @ err                           # shape (d,)
            grad_b = err.sum()
            return np.concatenate([grad_w, [grad_b]])

        x0 = np.zeros(n_features + 1)
        result = minimize(_neg_log_likelihood, x0, jac=_grad, method="L-BFGS-B")
        if not result.success:
            # Fall back to the OLS-on-binary form rather than failing hard;
            # warn so the caller knows the fit was not MLE.
            warnings.warn(
                f"LogisticTransfer MLE fit did not converge: {result.message}. "
                "Falling back to OLS-on-binary (legacy v0.1.0 behaviour).",
                stacklevel=2,
            )
            X_aug = np.column_stack([X, np.ones(len(X))])
            solution, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            self.weights = solution[:n_features]
            self.bias = float(solution[n_features])
        else:
            self.weights = result.x[:n_features]
            self.bias = float(result.x[n_features])
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

        features = self._extract_features(comparison)
        linear_pred = self.weights @ features + self.bias
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
