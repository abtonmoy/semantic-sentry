"""Calibration profile storage for transfer functions."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from semantic_sentry.transfer.function import TRANSFER_FEATURE_NAMES, LinearTransfer


@dataclass
class CalibrationProfile:
    """Calibration profile for a model family.

    Stores fitted transfer function parameters and metadata for reuse.

    Attributes:
        profile_name: Name of the calibration profile
        model_family: Model family identifier (e.g., "bert-base", "clip-vit")
        weights: Transfer function weights [w_cka, w_nps, w_iso]
        bias: Transfer function bias
        r_squared: Model fit quality
        n_samples: Number of calibration samples
        feature_names: Names of features used (defaults to the standard
            (1-cka, 1-nps, |isotropy_delta|) trio).
        clip: Whether the source LinearTransfer was constructed with
            ``clip=True`` (the non-negative-degradation contract — output
            clamped to ``[0, 1]``, negative targets rejected at fit-time).
            Persisted into the JSON so round-trips reconstruct the same
            contract. Defaults to ``False`` so v0.1.0 profiles load as the
            new signed default.
    """
    profile_name: str
    model_family: str
    weights: list[float]
    bias: float
    r_squared: float
    n_samples: int
    feature_names: list[str] = field(default_factory=lambda: list(TRANSFER_FEATURE_NAMES))
    clip: bool = False

    def to_transfer_function(self) -> LinearTransfer:
        """Convert profile to fitted LinearTransfer.

        The reconstructed transfer is constructed with the same ``clip``
        contract as the source — i.e. round-tripping a ``clip=True``
        transfer through a CalibrationProfile and back yields another
        ``clip=True`` transfer, so neither half of the clamp-output /
        reject-negative-input pair can silently drift across serialization.

        Returns:
            Fitted LinearTransfer instance
        """
        transfer = LinearTransfer(clip=self.clip)
        transfer.weights = np.array(self.weights)
        transfer.bias = self.bias
        transfer.r_squared = self.r_squared
        transfer._fitted = True
        return transfer

    def save(self, path: str | Path) -> None:
        """Save calibration profile to JSON.

        Args:
            path: File path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "profile_name": self.profile_name,
            "model_family": self.model_family,
            "weights": self.weights,
            "bias": self.bias,
            "r_squared": self.r_squared,
            "n_samples": self.n_samples,
            "feature_names": self.feature_names,
            "clip": self.clip,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationProfile":
        """Load calibration profile from JSON.

        Args:
            path: File path to load from

        Returns:
            Loaded CalibrationProfile. v0.1.0 profiles that predate the
            ``clip`` field load as ``clip=False`` (signed default) — under
            the v0.1.0 implementation they were silently clamped at
            predict-time regardless, so this is a behavioral upgrade for
            legacy profiles, not a regression.
        """
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        return cls(
            profile_name=data["profile_name"],
            model_family=data["model_family"],
            weights=data["weights"],
            bias=data["bias"],
            r_squared=data["r_squared"],
            n_samples=data["n_samples"],
            feature_names=data.get("feature_names", ["1-cka", "1-nps", "|isotropy_delta|"]),
            clip=data.get("clip", False),
        )

    @classmethod
    def from_transfer_function(
        cls,
        transfer: LinearTransfer,
        profile_name: str,
        model_family: str,
        n_samples: int
    ) -> "CalibrationProfile":
        """Create profile from fitted transfer function.

        Args:
            transfer: Fitted LinearTransfer
            profile_name: Name for the profile
            model_family: Model family identifier
            n_samples: Number of calibration samples

        Returns:
            CalibrationProfile carrying the source transfer's ``clip``
            contract so round-trips preserve the clamp-output /
            reject-negative-input pairing.
        """
        if not transfer._fitted:
            raise ValueError("Transfer function must be fitted")

        return cls(
            profile_name=profile_name,
            model_family=model_family,
            weights=transfer.weights.tolist(),
            bias=transfer.bias,
            r_squared=transfer.r_squared or 0.0,
            n_samples=n_samples,
            clip=transfer._clip,
        )


class CalibrationProfileStore:
    """Store for managing multiple calibration profiles."""

    def __init__(self, storage_dir: str | Path = "~/.semantic_sentry/calibration"):
        """Initialize profile store.

        Args:
            storage_dir: Directory to store profiles
        """
        self._storage_dir = Path(storage_dir).expanduser()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, CalibrationProfile] = {}

    def save(self, profile: CalibrationProfile) -> None:
        """Save a profile to storage.

        Args:
            profile: Profile to save
        """
        path = self._storage_dir / f"{profile.profile_name}.json"
        profile.save(path)
        self._cache[profile.profile_name] = profile

    def load(self, profile_name: str) -> CalibrationProfile:
        """Load a profile from storage.

        Args:
            profile_name: Name of the profile

        Returns:
            Loaded profile

        Raises:
            FileNotFoundError: If profile not found
        """
        # Check cache first
        if profile_name in self._cache:
            return self._cache[profile_name]

        # Load from disk
        path = self._storage_dir / f"{profile_name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Profile '{profile_name}' not found")

        profile = CalibrationProfile.load(path)
        self._cache[profile_name] = profile
        return profile

    def list_profiles(self) -> list[str]:
        """List available profiles.

        Returns:
            List of profile names
        """
        return [p.stem for p in self._storage_dir.glob("*.json")]

    def delete(self, profile_name: str) -> None:
        """Delete a profile.

        Args:
            profile_name: Name of profile to delete
        """
        path = self._storage_dir / f"{profile_name}.json"
        if path.exists():
            path.unlink()

        if profile_name in self._cache:
            del self._cache[profile_name]
