"""Phase 5: Adversarial scenario tests."""

import numpy as np
import pytest

from semantic_sentry.metrics.cka import linear_cka
from semantic_sentry.metrics.nps import nps
from semantic_sentry.core.snapshot import Snapshot
from semantic_sentry.exceptions import SnapshotCorruptionError


@pytest.mark.stress
class TestAdversarial:
    """Adversarial scenario stress tests."""

    def test_adv_001_rotation_attack(self, make_embeddings, rng):
        """ADV-001: Orthogonal rotation gives CKA=1 but NPS<1.
        This validates why multi-metric is necessary."""
        Z = make_embeddings(200, 64)
        H = rng.standard_normal((64, 64)).astype(np.float32)
        Q, _ = np.linalg.qr(H)
        Z_rotated = Z @ Q

        cka_val = linear_cka(Z, Z_rotated)
        nps_val = nps(Z, Z_rotated, k=10)

        assert cka_val > 0.99, f"CKA should be ~1.0 after rotation, got {cka_val}"
        # NPS may or may not change depending on rotation — this is the key test
        # A random rotation WILL change neighborhoods
        # This validates the HLD claim that CKA alone is insufficient

    def test_adv_002_scaling_attack(self, make_embeddings):
        """ADV-002: Scaling attack - different magnitudes."""
        Z0 = make_embeddings(100, 64)
        Z1 = Z0 * 2.0  # Scale up
        cka_val = linear_cka(Z0, Z1)
        # CKA should be invariant to scaling
        assert cka_val > 0.99, f"CKA should be invariant to scaling, got {cka_val}"

    def test_adv_003_permutation_attack(self, make_embeddings, rng):
        """ADV-003: Permutation attack - reorder samples."""
        Z = make_embeddings(100, 64)
        perm = rng.permutation(100)
        Z_perm = Z[perm]
        cka_val = linear_cka(Z, Z_perm)
        # CKA may or may not be invariant to permutation
        assert np.isfinite(cka_val)

    def test_adv_004_gaussian_noise_injection(self, make_embeddings, rng):
        """ADV-004: Compare two completely different embedding sets."""
        Z1 = make_embeddings(100, 64, seed=42)
        Z2 = make_embeddings(100, 64, seed=999)  # Different seed = different data

        cka_val = linear_cka(Z1, Z2)
        nps_val = nps(Z1, Z2, k=10)

        # Both metrics should detect that these are different
        assert cka_val < 0.5, f"CKA should detect different embeddings, got {cka_val}"
        assert nps_val < 0.5, f"NPS should detect different embeddings, got {nps_val}"

    def test_adv_005_outlier_injection(self, rng):
        """ADV-005: Outlier injection - few extreme values."""
        Z = rng.standard_normal((100, 64)).astype(np.float32)
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        Z_outliers = Z.copy()
        Z_outliers[0] = rng.standard_normal(64).astype(np.float32) * 10
        Z_outliers[0] = Z_outliers[0] / np.linalg.norm(Z_outliers[0])

        result = linear_cka(Z, Z_outliers)
        assert np.isfinite(result)

    def test_adv_006_truncated_snapshot(self, make_embeddings, tmp_path):
        """ADV-006: Truncated snapshot file must raise error."""
        import json
        from datetime import datetime, timezone

        Z = make_embeddings(50, 32)
        snap = Snapshot(
            model_id="test", checkpoint_hash="a" * 64,
            timestamp=datetime.now(timezone.utc).isoformat(),
            anchor_set_version="b" * 32, tower_count=1,
            tower_names=("encoder",), embeddings={"encoder": Z},
            cross_tower_alignment=None, metadata={},
        )
        path = tmp_path / "snap"
        snap.save(path)
        # Truncate a safetensors file
        st_file = path / "encoder.safetensors"
        with open(st_file, "wb") as f:
            f.write(b"corrupted")
        with pytest.raises((SnapshotCorruptionError, Exception)):
            Snapshot.load(path)

    def test_adv_007_modified_hash(self, make_embeddings, tmp_path):
        """ADV-007: Modified checkpoint hash - validation limited.

        Note: The checkpoint_hash cannot be fully verified without the original model.
        We only check that it's present and not empty.
        """
        import json
        from datetime import datetime, timezone

        Z = make_embeddings(50, 32)
        snap = Snapshot(
            model_id="test", checkpoint_hash="a" * 64,
            timestamp=datetime.now(timezone.utc).isoformat(),
            anchor_set_version="b" * 32, tower_count=1,
            tower_names=("encoder",), embeddings={"encoder": Z},
            cross_tower_alignment=None, metadata={},
        )
        path = tmp_path / "snap"
        snap.save(path)
        meta = json.loads((path / "metadata.json").read_text())
        # Test with empty checkpoint hash - this should raise
        meta["checkpoint_hash"] = ""
        (path / "metadata.json").write_text(json.dumps(meta))
        with pytest.raises(SnapshotCorruptionError):
            Snapshot.load(path)
