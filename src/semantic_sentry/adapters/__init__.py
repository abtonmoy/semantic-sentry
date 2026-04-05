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
    1. CLIPAdapter (has encode_image + encode_text)
    2. SentenceTransformerAdapter (isinstance check)
    3. ONNXAdapter (isinstance InferenceSession)
    4. HuggingFaceAdapter (isinstance PreTrainedModel)
    5. Raise AdapterDetectionError
    
    Args:
        model: Model instance to detect adapter for
        
    Returns:
        Appropriate EncoderAdapter instance
        
    Raises:
        AdapterDetectionError: If no adapter matches the model type
    """
    from semantic_sentry.exceptions import AdapterDetectionError

    # Check for CLIP (OpenCLIP)
    if CLIPAdapter is not None:
        if hasattr(model, 'encode_image') and hasattr(model, 'encode_text'):
            raise AdapterDetectionError(
                "CLIP model detected but requires tokenizer and preprocess. "
                "Use CLIPAdapter directly with: CLIPAdapter(model, tokenizer, preprocess)"
            )

    # Check for SentenceTransformer
    if SentenceTransformerAdapter is not None:
        try:
            from sentence_transformers import SentenceTransformer
            if isinstance(model, SentenceTransformer):
                return SentenceTransformerAdapter(model)
        except ImportError:
            pass

    # Check for ONNX
    if ONNXAdapter is not None:
        try:
            import onnxruntime as ort
            if isinstance(model, ort.InferenceSession):
                return ONNXAdapter(model)
        except ImportError:
            pass

    # Check for HuggingFace
    if HuggingFaceAdapter is not None:
        try:
            from transformers import PreTrainedModel
            if isinstance(model, PreTrainedModel):
                raise AdapterDetectionError(
                    "HuggingFace model detected but requires tokenizer. "
                    "Use HuggingFaceAdapter directly with: HuggingFaceAdapter(model, tokenizer)"
                )
        except ImportError:
            pass

    # No adapter found
    model_type = type(model).__name__
    raise AdapterDetectionError(
        f"No adapter found for model type '{model_type}'. "
        f"Available adapters: CustomAdapter. "
        f"Consider using CustomAdapter with a custom encode function."
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

