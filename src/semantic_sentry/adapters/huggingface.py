"""HuggingFace Transformers adapter."""

from typing import Any

import torch

try:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from semantic_sentry.adapters.base import SingleTowerAdapter


class HuggingFaceAdapter(SingleTowerAdapter):
    """Adapter for HuggingFace Transformers models.
    
    This adapter handles encoder-only models like BERT, RoBERTa, E5, etc.
    It extracts the [CLS] token representation or mean pools if specified.
    
    Example:
        from transformers import AutoModel, AutoTokenizer
        
        model = AutoModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        adapter = HuggingFaceAdapter(model, tokenizer)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        pooling: str = "cls",
        normalize: bool = True
    ):
        """Initialize HuggingFace adapter.
        
        Args:
            model: HuggingFace model instance
            tokenizer: HuggingFace tokenizer instance
            pooling: Pooling strategy ('cls', 'mean', or 'last')
            normalize: Whether to L2-normalize embeddings
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required. "
                "Install with: pip install semantic-sentry[dev]"
            )

        self._model = model
        self._tokenizer = tokenizer
        self._pooling = pooling
        self._normalize = normalize
        self._device = next(model.parameters()).device

    def encode(self, inputs: Any) -> dict[str, torch.Tensor]:
        """Encode text inputs into embeddings.
        
        Args:
            inputs: Text inputs (str, list[str], or pre-tokenized dict)
            
        Returns:
            Dict with single tower 'encoder' containing embeddings
        """
        # Tokenize if needed
        if isinstance(inputs, str):
            inputs = [inputs]

        if isinstance(inputs, list):
            inputs = self._tokenizer(
                inputs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )

        # Move to model device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self._model(**inputs)
            hidden_states = outputs.last_hidden_state

        # Pool embeddings
        if self._pooling == "cls":
            # Use [CLS] token (first token)
            embeddings = hidden_states[:, 0, :]
        elif self._pooling == "mean":
            # Mean pooling (excluding padding)
            attention_mask = inputs["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).float()
            embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            embeddings = embeddings / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        elif self._pooling == "last":
            # Use last non-padding token
            attention_mask = inputs["attention_mask"]
            last_positions = attention_mask.sum(dim=1) - 1
            embeddings = hidden_states[torch.arange(len(last_positions)), last_positions]
        else:
            raise ValueError(f"Unknown pooling: {self._pooling}")

        # Normalize
        if self._normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return {self.tower_name: embeddings}

    @property
    def tower_name(self) -> str:
        """Name of the single tower."""
        return "encoder"
