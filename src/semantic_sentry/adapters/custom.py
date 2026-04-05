"""Custom adapter for user-defined encode functions."""

from collections.abc import Callable
from typing import Any

import torch

from semantic_sentry.adapters.base import EncoderAdapter


class CustomAdapter(EncoderAdapter):
    """Adapter for custom encode functions.
    
    This adapter allows users to provide their own encode function
    for models not supported by built-in adapters.
    
    Example:
        def my_encode(inputs):
            # Custom encoding logic
            embeddings = my_model.encode(inputs)
            return {"encoder": torch.tensor(embeddings)}
        
        adapter = CustomAdapter(encode_fn=my_encode, tower_count=1)
    """

    def __init__(
        self,
        encode_fn: Callable[[Any], dict[str, torch.Tensor] | torch.Tensor],
        tower_count: int = 1,
        tower_names: list[str] | None = None,
        normalize: bool = True
    ):
        """Initialize custom adapter.
        
        Args:
            encode_fn: Function that takes inputs and returns embeddings
            tower_count: Number of towers
            tower_names: Optional list of tower names
            normalize: Whether to L2-normalize embeddings
        """
        self._encode_fn = encode_fn
        self._tower_count = tower_count
        self._tower_names = tower_names or [f"tower_{i}" for i in range(tower_count)]
        self._normalize = normalize

    def encode(self, inputs: Any) -> dict[str, torch.Tensor]:
        """Encode inputs using the custom function.
        
        Args:
            inputs: Input data
            
        Returns:
            Dict mapping tower name to embedding tensor
        """
        result = self._encode_fn(inputs)

        # Handle single tensor output
        if isinstance(result, torch.Tensor):
            result = {self._tower_names[0]: result}

        # Normalize if requested
        if self._normalize:
            result = {
                name: self._normalize_tensor(tensor)
                for name, tensor in result.items()
            }

        return result

    def list_towers(self) -> list[str]:
        """Return tower names."""
        return self._tower_names

    @property
    def tower_count(self) -> int:
        """Number of towers."""
        return self._tower_count

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """L2 normalize a tensor."""
        return torch.nn.functional.normalize(tensor, p=2, dim=1)
