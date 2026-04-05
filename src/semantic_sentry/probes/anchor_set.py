"""Anchor set dataclass for probe management."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AnchorSet:
    """Immutable anchor set for drift detection probes.
    
    Attributes:
        inputs: The input data (text, images, etc.) for the anchor set
        labels: Optional labels for the anchor points
        version_hash: Deterministic hash computed from serialized inputs
        modality: The modality of the data (e.g., 'text', 'image', 'multimodal')
        n_samples: Number of samples in the anchor set
    """
    inputs: Any
    labels: tuple = field(default_factory=tuple)
    modality: str = "text"
    version_hash: str = field(default="", repr=False)
    n_samples: int = field(default=0)

    def __post_init__(self):
        """Compute version_hash and n_samples if not provided."""
        if not self.version_hash:
            # Compute deterministic hash from serialized inputs
            hash_val = self._compute_hash(self.inputs)
            object.__setattr__(self, 'version_hash', hash_val)

        if self.n_samples == 0:
            # Infer n_samples from inputs
            n = self._infer_n_samples(self.inputs)
            object.__setattr__(self, 'n_samples', n)

    @staticmethod
    def _compute_hash(inputs: Any) -> str:
        """Compute deterministic hash from inputs."""
        try:
            # Try to serialize to JSON
            if isinstance(inputs, (list, tuple)):
                serialized = json.dumps(inputs, sort_keys=True)
            elif isinstance(inputs, str):
                serialized = inputs
            elif hasattr(inputs, 'tolist'):
                # numpy array or similar
                serialized = json.dumps(inputs.tolist())
            else:
                serialized = str(inputs)
            return hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:16]
        except (TypeError, ValueError):
            # Fallback: use string representation
            return hashlib.sha256(str(inputs).encode('utf-8')).hexdigest()[:16]

    @staticmethod
    def _infer_n_samples(inputs: Any) -> int:
        """Infer number of samples from inputs."""
        if hasattr(inputs, '__len__'):
            return len(inputs)
        if hasattr(inputs, 'shape'):
            return inputs.shape[0]
        return 0
