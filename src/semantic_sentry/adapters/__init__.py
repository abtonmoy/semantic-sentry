"""Encoder adapters for different model types."""

from semantic_sentry.adapters.base import EncoderAdapter
from semantic_sentry.adapters.custom import CustomAdapter

# Optional adapters - import if available
try:
    from semantic_sentry.adapters.huggingface import HuggingFaceAdapter
except ImportError:
    HuggingFaceAdapter = None  # type: ignore

try:
    from semantic_sentry.adapters.clip import CLIPAdapter
except ImportError:
    CLIPAdapter = None  # type: ignore

try:
    from semantic_sentry.adapters.sentence_transformer import SentenceTransformerAdapter
except ImportError:
    SentenceTransformerAdapter = None  # type: ignore

try:
    from semantic_sentry.adapters.onnx_adapter import ONNXAdapter
except ImportError:
    ONNXAdapter = None  # type: ignore


def detect_adapter(model) -> EncoderAdapter:
    """Auto-detect the appropriate adapter for a model.

    Priority order:
    1. SentenceTransformerAdapter (isinstance check)
    2. ONNXAdapter (isinstance InferenceSession)

    Adapter types that require extra constructor args (CLIP needs a
    tokenizer + preprocess; HuggingFace `PreTrainedModel` needs a
    tokenizer) cannot be returned from auto-detection — pass them to
    ``CLIPAdapter`` / ``HuggingFaceAdapter`` directly. We log a debug
    message when one of those types is detected so the user knows why
    auto-detection didn't match, but we don't raise on that case;
    auto-detection raises only when no adapter at all matches.

    Args:
        model: Model instance to detect adapter for

    Returns:
        Appropriate EncoderAdapter instance.

    Raises:
        AdapterDetectionError: If no adapter at all matches the model type.
    """
    import logging

    from semantic_sentry.exceptions import AdapterDetectionError

    log = logging.getLogger(__name__)

    # CLIP — needs tokenizer + preprocess; cannot be auto-constructed.
    if (
        CLIPAdapter is not None
        and hasattr(model, 'encode_image')
        and hasattr(model, 'encode_text')
    ):
        log.debug(
            "detect_adapter: model looks like a CLIP model but CLIPAdapter "
            "needs a tokenizer + preprocess; skipping auto-detection. "
            "Use CLIPAdapter(model, tokenizer, preprocess) directly."
        )

    # SentenceTransformer — fully constructible from the model alone.
    if SentenceTransformerAdapter is not None:
        try:
            from sentence_transformers import SentenceTransformer
            if isinstance(model, SentenceTransformer):
                return SentenceTransformerAdapter(model)
        except ImportError:
            pass

    # ONNX — fully constructible from the model alone.
    if ONNXAdapter is not None:
        try:
            import onnxruntime as ort
            if isinstance(model, ort.InferenceSession):
                return ONNXAdapter(model)
        except ImportError:
            pass

    # HuggingFace — needs a tokenizer; cannot be auto-constructed.
    if HuggingFaceAdapter is not None:
        try:
            from transformers import PreTrainedModel
            if isinstance(model, PreTrainedModel):
                log.debug(
                    "detect_adapter: model looks like a HuggingFace "
                    "PreTrainedModel but HuggingFaceAdapter needs a tokenizer; "
                    "skipping auto-detection. Use "
                    "HuggingFaceAdapter(model, tokenizer) directly."
                )
        except ImportError:
            pass

    # No adapter matched.
    model_type = type(model).__name__
    raise AdapterDetectionError(
        f"No adapter found for model type '{model_type}'. "
        f"Available adapters: CustomAdapter. "
        f"Consider using CustomAdapter with a custom encode function, "
        f"or pass CLIPAdapter / HuggingFaceAdapter explicitly if your "
        f"model needs a tokenizer / preprocess."
    )


__all__ = [
    "EncoderAdapter",
    "CustomAdapter",
    "detect_adapter",
]

# Add optional adapters to __all__ if available
if HuggingFaceAdapter is not None:
    __all__.append("HuggingFaceAdapter")
if CLIPAdapter is not None:
    __all__.append("CLIPAdapter")
if SentenceTransformerAdapter is not None:
    __all__.append("SentenceTransformerAdapter")
if ONNXAdapter is not None:
    __all__.append("ONNXAdapter")

