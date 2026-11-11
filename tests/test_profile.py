"""Tests for the deck profile."""

import numpy as np
import pytest
import scipy.sparse as sp

from src.profile import build_profile, copy_weights


def test_copy_weights_are_capped() -> None:
    assert np.allclose(copy_weights([1, 2, 4, 8], 4), [0.25, 0.5, 1.0, 1.0])


def test_profile_is_unit_norm_weighted_mean() -> None:
    X = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]))
    p = build_profile(X, [0, 1], [1.0, 0.25])
    expected = np.array([1.0, 0.25]) / np.linalg.norm([1.0, 0.25])
    assert np.allclose(p, expected)
    assert np.isclose(np.linalg.norm(p), 1.0)


def test_profile_uses_only_given_rows() -> None:
    X = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert np.allclose(build_profile(X, [0], [1.0]), [1.0, 0.0])


def test_profile_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        build_profile(sp.csr_matrix((2, 2)), [], [])
