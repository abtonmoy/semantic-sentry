"""Theoretical NPS bounds on downstream task performance."""

import numpy as np


class NPSBound:
    """Theoretical bounds on downstream task degradation from NPS drop.

    These bounds provide theoretical guarantees on worst-case degradation
    based on the Neighborhood Preservation Score.
    """

    @staticmethod
    def lower_bound(nps_score: float) -> float:
        """Theoretical lower bound on retrieval degradation from NPS drop.

        The bound is: degradation >= 1 - NPS

        Args:
            nps_score: NPS score in [0, 1]

        Returns:
            Lower bound on degradation in [0, 1]
        """
        delta = 1.0 - nps_score
        return max(0.0, delta)

    @staticmethod
    def upper_bound(nps_score: float, k: int = 10) -> float:
        """Theoretical upper bound on retrieval degradation.

        This is a looser bound that accounts for the k-NN structure.

        Args:
            nps_score: NPS score in [0, 1]
            k: Number of neighbors considered

        Returns:
            Upper bound on degradation in [0, 1]
        """
        delta = 1.0 - nps_score
        # Upper bound scales with k
        return min(1.0, delta * (1.0 + np.log(k)))

    @staticmethod
    def confidence_interval(
        nps_score: float,
        n_samples: int,
        confidence: float = 0.95
    ) -> tuple[float, float]:
        """Compute confidence interval for NPS-based degradation bound.

        Args:
            nps_score: Observed NPS score
            nps_samples: Number of samples used to compute NPS
            confidence: Confidence level (default: 0.95)

        Returns:
            (lower_bound, upper_bound) tuple
        """
        # Point estimate
        point_est = 1.0 - nps_score

        # Standard error (binomial approximation)
        se = np.sqrt(nps_score * (1 - nps_score) / n_samples) if n_samples > 0 else 0.0

        # Confidence interval (using normal approximation)
        z = 1.96 if confidence == 0.95 else 2.58  # 95% or 99%
        margin = z * se

        lower = max(0.0, point_est - margin)
        upper = min(1.0, point_est + margin)

        return lower, upper


def retrieval_recall_at_k_bound(
    nps_score: float,
    baseline_recall: float,
    k: int = 10
) -> float:
    """Bound on retrieval recall given NPS score.

    Args:
        nps_score: NPS score
        baseline_recall: Baseline recall@k on clean data
        k: Number of neighbors

    Returns:
        Lower bound on recall@k after drift
    """
    # NPS gives overlap fraction
    # Lower bound on recall is baseline_recall * nps_score
    return baseline_recall * nps_score


def classification_accuracy_bound(
    nps_score: float,
    baseline_accuracy: float
) -> float:
    """Bound on classification accuracy given NPS score.

    Args:
        nps_score: NPS score
        baseline_accuracy: Baseline accuracy on clean data

    Returns:
        Lower bound on accuracy after drift
    """
    # Conservative bound: accuracy can't drop below baseline * NPS
    # This assumes worst-case where all mismatched neighbors are wrong
    return baseline_accuracy * nps_score
