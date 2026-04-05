"""SentenceTransformer adapter."""

from typing import Any

import torch

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from semantic_sentry.adapters.base import SingleTowerAdapter


class SentenceTransformerAdapter(SingleTowerAdapter):
    """Adapter for SentenceTransformer models.
    
    This adapter handles sentence embedding models like all-MiniLM-L6-v2,
    all-mpnet-base-v2, etc.
    
    Example:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        adapter = SentenceTransformerAdapter(model)
    """

    def __init__(
        self,
        model: Any,
        normalize: bool = True
    ):
        """Initialize SentenceTransformer adapter.
        
        Args:
            model: SentenceTransformer model instance
            normalize: Whether to L2-normalize embeddings
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers library required. "
                "Install with: pip install semantic-sentry[sentence-transformers]"
            )

        self._model = model
        self._normalize = normalize

    def encode(self, inputs: Any) -> dict[str, torch.Tensor]:
        """Encode text inputs into embeddings.
        
        Args:
            inputs: Text inputs (str or list[str])
            
        Returns:
            Dict with single tower 'encoder' containing embeddings
        """
        # Handle single string
        if isinstance(inputs, str):
            inputs = [inputs]

        # Encode using SentenceTransformer
        embeddings = self._model.encode(
            inputs,
            convert_to_tensor=True,
            normalize_embeddings=self._normalize
        )

        return {self.tower_name: embeddings}

    @property
    def tower_name(self) -> str:
        """Name of the single tower."""
        return "encoder"
