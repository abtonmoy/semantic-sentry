"""ONNX Runtime adapter."""

from typing import Any

import torch

try:
    import onnxruntime as ort  # noqa: F401 — re-imported lazily where needed
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from semantic_sentry.adapters.base import EncoderAdapter


class ONNXAdapter(EncoderAdapter):
    """Adapter for ONNX Runtime inference sessions.

    This adapter handles ONNX models for efficient inference.
    Supports both CPU and GPU execution providers.

    Example:
        import onnxruntime as ort

        session = ort.InferenceSession("model.onnx")
        adapter = ONNXAdapter(session, input_name="input", output_name="output")
    """

    def __init__(
        self,
        session: Any,
        input_name: str = "input",
        output_names: list[str] | None = None,
        normalize: bool = True
    ):
        """Initialize ONNX adapter.

        Args:
            session: ONNX Runtime InferenceSession
            input_name: Name of the input node
            output_names: Names of output nodes (inferred if None)
            normalize: Whether to L2-normalize embeddings
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime library required. "
                "Install with: pip install semantic-sentry[onnx]"
            )

        self._session = session
        self._input_name = input_name
        self._normalize = normalize

        # Infer output names if not provided
        if output_names is None:
            output_names = [out.name for out in session.get_outputs()]
        self._output_names = output_names

    def encode(self, inputs: Any) -> dict[str, torch.Tensor]:
        """Encode inputs using ONNX Runtime.

        Args:
            inputs: Input tensor or numpy array

        Returns:
            Dict mapping output name to embedding tensor
        """
        # Convert to numpy if needed
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu().numpy()

        # Ensure correct shape (add batch dimension if needed)
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)

        # Run inference
        outputs = self._session.run(self._output_names, {self._input_name: inputs})

        # Convert to tensors and normalize
        result = {}
        for name, output in zip(self._output_names, outputs, strict=False):
            tensor = torch.from_numpy(output)
            if self._normalize:
                tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)
            result[name] = tensor

        return result

    def list_towers(self) -> list[str]:
        """Return output node names as tower names."""
        return self._output_names

    @property
    def tower_count(self) -> int:
        """Number of output nodes."""
        return len(self._output_names)
