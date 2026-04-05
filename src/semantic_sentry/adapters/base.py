"""Abstract base class for encoder adapters."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class EncoderAdapter(ABC):
    """Abstract base class for encoder adapters.
    
    An adapter wraps a model and provides a unified interface for encoding
    inputs into embeddings. Different adapters handle different model types
    (HuggingFace Transformers, CLIP, SentenceTransformers, etc.).
    """

    @abstractmethod
    def encode(self, inputs: Any) -> dict[str, torch.Tensor]:
        """Encode inputs into embeddings for all towers.
        
        Args:
            inputs: Input data (format depends on model type)
            
        Returns:
            Dict mapping tower name to normalized embedding tensor of shape (n, d)
        """
        pass

    @abstractmethod
    def list_towers(self) -> list[str]:
        """Return ordered list of tower names.
        
        Returns:
            List of tower names
        """
        pass

    @property
    def tower_count(self) -> int:
        """Number of towers in the model."""
        return len(self.list_towers())

    def encode_numpy(self, inputs: Any) -> dict[str, np.ndarray]:
        """Encode inputs and return as numpy arrays.
        
        Args:
            inputs: Input data
            
        Returns:
            Dict mapping tower name to normalized embedding array of shape (n, d)
        """
        tensors = self.encode(inputs)
        return {
            name: tensor.cpu().numpy()
            for name, tensor in tensors.items()
        }

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """L2 normalize embeddings.
        
        Args:
            tensor: Embedding tensor of shape (n, d)
            
        Returns:
            L2-normalized tensor
        """
        return torch.nn.functional.normalize(tensor, p=2, dim=1)


class SingleTowerAdapter(EncoderAdapter):
    """Base class for single-tower adapters."""

    @property
    def tower_name(self) -> str:
        """Name of the single tower."""
        return "encoder"

    def list_towers(self) -> list[str]:
        """Single tower returns ['encoder']."""
        return [self.tower_name]


class MultiTowerAdapter(EncoderAdapter):
    """Base class for multi-tower adapters."""

    def __init__(self, tower_names: list[str]) -> None:
        """Initialize with tower names.
        
        Args:
            tower_names: Ordered list of tower names
        """
        self._tower_names = tower_names

    def list_towers(self) -> list[str]:
        """Return configured tower names."""
        return self._tower_names
