"""Parity tests: torch metric backend vs the numpy implementations."""

import numpy as np
import pytest
import torch

from semantic_sentry.metrics.cka import linear_cka
from semantic_sentry.metrics.isotropy import isotropy_delta
from semantic_sentry.metrics.nps import nps
from semantic_sentry.metrics.torch_backend import (
    compute_drift_metrics_torch,
    isotropy_delta_torch,
    linear_cka_torch,
    nps_torch,
)


@pytest.fixture
def pair():
    rng = np.random.default_rng(0)
    Z0 = rng.standard_normal((80, 32)).astype(np.float32)
    # Z1 = Z0 plus structured noise so metrics land off the extremes.
    Z1 = (Z0 + 0.4 * rng.standard_normal((80, 32))).astype(np.float32)
    return Z0, Z1


def test_cka_parity(pair):
    Z0, Z1 = pair
    np_val = linear_cka(Z0, Z1)
    th_val = linear_cka_torch(torch.from_numpy(Z0), torch.from_numpy(Z1))
    assert np_val == pytest.approx(th_val, abs=1e-5)


def test_isotropy_parity(pair):
    Z0, Z1 = pair
    np_val = isotropy_delta(Z0, Z1)
    th_val = isotropy_delta_torch(torch.from_numpy(Z0), torch.from_numpy(Z1))
    assert np_val == pytest.approx(th_val, abs=1e-5)


def test_nps_parity(pair):
    Z0, Z1 = pair
    np_val = nps(Z0, Z1, k=10)
    th_val = nps_torch(torch.from_numpy(Z0), torch.from_numpy(Z1), k=10)
    # NPS is a set-overlap fraction; both paths exclude self deterministically.
    assert np_val == pytest.approx(th_val, abs=1e-6)


def test_identical_inputs_are_perfect():
    Z = torch.randn(50, 16)
    assert linear_cka_torch(Z, Z) == pytest.approx(1.0, abs=1e-5)
    assert nps_torch(Z, Z, k=5) == pytest.approx(1.0, abs=1e-6)
    assert isotropy_delta_torch(Z, Z) == pytest.approx(0.0, abs=1e-6)


def test_compute_all_keys(pair):
    Z0, Z1 = pair
    out = compute_drift_metrics_torch(torch.from_numpy(Z0), torch.from_numpy(Z1))
    assert set(out) == {"cka", "nps", "isotropy_delta"}
    assert all(isinstance(v, float) for v in out.values())


def test_nps_degenerate_returns_one():
    Z = torch.randn(5, 8)
    assert nps_torch(Z, Z, k=10) == 1.0
