"""CLIP adapter for vision-language models."""

from typing import Any

import torch

try:
    import open_clip  # noqa: F401 — import is the availability probe
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

from semantic_sentry.adapters.base import MultiTowerAdapter


class CLIPAdapter(MultiTowerAdapter):
    """Adapter for CLIP models (OpenCLIP).

    This adapter handles dual-tower vision-language models.
    It separately encodes images and text into a shared embedding space.

    Example:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        adapter = CLIPAdapter(model, tokenizer, preprocess)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        preprocess: Any,
        normalize: bool = True
    ):
        """Initialize CLIP adapter.

        Args:
            model: CLIP model instance
            tokenizer: CLIP tokenizer
            preprocess: Image preprocessing function
            normalize: Whether to L2-normalize embeddings
        """
        if not CLIP_AVAILABLE:
            raise ImportError(
                "open-clip-torch library required. "
                "Install with: pip install semantic-sentry[clip]"
            )

        super().__init__(tower_names=["vision", "language"])
        self._model = model
        self._tokenizer = tokenizer
        self._preprocess = preprocess
        self._normalize = normalize
        self._device = next(model.parameters()).device

    def encode(self, inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Encode inputs into embeddings.

        Args:
            inputs: Dict with keys 'images' and/or 'texts'
                - 'images': List of PIL Images or preprocessed tensors
                - 'texts': List of text strings

        Returns:
            Dict with towers 'vision' and/or 'language' containing embeddings
        """
        embeddings = {}

        # Encode images if provided
        if "images" in inputs:
            embeddings["vision"] = self._encode_images(inputs["images"])

        # Encode text if provided
        if "texts" in inputs:
            embeddings["language"] = self._encode_text(inputs["texts"])

        return embeddings

    def _encode_images(self, images: Any) -> torch.Tensor:
        """Encode images into vision embeddings."""
        # Preprocess if needed
        if hasattr(images, '__iter__') and not isinstance(images, torch.Tensor):
            # Assume PIL Images
            images = torch.stack([self._preprocess(img) for img in images])

        if isinstance(images, list):
            images = torch.stack(images)

        images = images.to(self._device)

        with torch.no_grad():
            embeddings = self._model.encode_image(images)

        if self._normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text into language embeddings."""
        # Tokenize
        tokens = self._tokenizer(texts).to(self._device)

        with torch.no_grad():
            embeddings = self._model.encode_text(tokens)

        if self._normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def encode_images(self, images: Any) -> dict[str, torch.Tensor]:
        """Convenience method to encode only images.

        Args:
            images: List of PIL Images or preprocessed tensors

        Returns:
            Dict with 'vision' tower embeddings
        """
        return {"vision": self._encode_images(images)}

    def encode_text(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Convenience method to encode only text.

        Args:
            texts: List of text strings

        Returns:
            Dict with 'language' tower embeddings
        """
        return {"language": self._encode_text(texts)}
