"""Evaluation registry for downstream task evaluators."""

from abc import ABC, abstractmethod

import numpy as np

from semantic_sentry.core.snapshot import Snapshot
from semantic_sentry.probes.anchor_set import AnchorSet


class Evaluator(ABC):
    """Abstract base class for downstream task evaluators."""

    @abstractmethod
    def evaluate(
        self,
        snapshot: Snapshot,
        anchor_set: AnchorSet
    ) -> float:
        """Evaluate performance on a snapshot.
        
        Args:
            snapshot: Model snapshot
            anchor_set: Anchor set with labels
            
        Returns:
            Performance score (higher is better)
        """
        pass

    def evaluate_delta(
        self,
        snapshot_v0: Snapshot,
        snapshot_v1: Snapshot,
        anchor_set: AnchorSet
    ) -> float:
        """Evaluate performance delta between snapshots.
        
        Args:
            snapshot_v0: Base snapshot
            snapshot_v1: Updated snapshot
            anchor_set: Anchor set with labels
            
        Returns:
            Performance delta (negative means degradation)
        """
        perf_v0 = self.evaluate(snapshot_v0, anchor_set)
        perf_v1 = self.evaluate(snapshot_v1, anchor_set)
        return perf_v1 - perf_v0


class EvaluatorRegistry:
    """Registry for downstream task evaluators."""

    def __init__(self):
        """Initialize empty registry."""
        self._evaluators: dict[str, Evaluator] = {}

    def register(self, name: str, evaluator: Evaluator) -> None:
        """Register an evaluator.
        
        Args:
            name: Evaluator name
            evaluator: Evaluator instance
        """
        self._evaluators[name] = evaluator

    def get(self, name: str) -> Evaluator:
        """Get evaluator by name.
        
        Args:
            name: Evaluator name
            
        Returns:
            Evaluator instance
        """
        return self._evaluators[name]

    def list_evaluators(self) -> list[str]:
        """List registered evaluators.
        
        Returns:
            List of evaluator names
        """
        return list(self._evaluators.keys())


class RetrievalEvaluator(Evaluator):
    """Dense retrieval evaluator using Mean Reciprocal Rank (MRR)."""

    def __init__(self, k: int = 10):
        """Initialize retrieval evaluator.
        
        Args:
            k: Number of neighbors to consider
        """
        self.k = k

    def evaluate(
        self,
        snapshot: Snapshot,
        anchor_set: AnchorSet
    ) -> float:
        """Evaluate retrieval performance using MRR@k.
        
        For each query, find its position in k-NN results.
        
        Args:
            snapshot: Model snapshot
            anchor_set: Anchor set
            
        Returns:
            Mean Reciprocal Rank
        """
        if anchor_set.labels is None or len(anchor_set.labels) == 0:
            raise ValueError("Anchor set must have labels for evaluation")

        embeddings = snapshot.get_tower(snapshot.tower_names[0])
        labels = anchor_set.labels

        # Compute pairwise similarities
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        similarities = embeddings_norm @ embeddings_norm.T

        # For each point, check if same-label points are in top-k
        mrr_sum = 0.0
        n = len(labels)

        for i in range(n):
            # Get top-k neighbors (excluding self)
            top_k = np.argsort(-similarities[i])[1:self.k+1]

            # Find first same-label neighbor
            for rank, neighbor_idx in enumerate(top_k, start=1):
                if labels[neighbor_idx] == labels[i]:
                    mrr_sum += 1.0 / rank
                    break

        return mrr_sum / n if n > 0 else 0.0


class ClassificationEvaluator(Evaluator):
    """k-NN classification evaluator."""

    def __init__(self, k: int = 5):
        """Initialize classification evaluator.
        
        Args:
            k: Number of neighbors for k-NN
        """
        self.k = k

    def evaluate(
        self,
        snapshot: Snapshot,
        anchor_set: AnchorSet
    ) -> float:
        """Evaluate classification accuracy using k-NN.
        
        Args:
            snapshot: Model snapshot
            anchor_set: Anchor set with labels
            
        Returns:
            Classification accuracy
        """
        if anchor_set.labels is None or len(anchor_set.labels) == 0:
            raise ValueError("Anchor set must have labels for evaluation")

        embeddings = snapshot.get_tower(snapshot.tower_names[0])
        labels = np.array(anchor_set.labels)

        # Compute pairwise similarities
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        similarities = embeddings_norm @ embeddings_norm.T

        # For each point, predict label from k-NN (excluding self)
        correct = 0
        n = len(labels)

        for i in range(n):
            # Get top-k neighbors (excluding self)
            top_k = np.argsort(-similarities[i])[1:self.k+1]

            # Majority vote
            neighbor_labels = labels[top_k]
            predicted = max(set(neighbor_labels), key=lambda x: np.sum(neighbor_labels == x))

            if predicted == labels[i]:
                correct += 1

        return correct / n if n > 0 else 0.0
