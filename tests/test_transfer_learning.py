# -*- coding: utf-8 -*-
# [METABCI-PYTEST-SUITE] Tests for transfer learning algorithms
#
# Tests for: metabci.brainda.algorithms.transfer_learning

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from metabci.brainda.algorithms.transfer_learning.lst import (
    lst_kernel,
    LST,
)


# ============================================================================
# Tests for LST kernel
# ============================================================================

class TestLstKernel:
    """Tests for the Least Squares Transformation kernel."""

    def test_output_shape(self):
        """Projection matrix should have correct shape."""
        n_features = 10
        S = np.random.RandomState(42).randn(5, n_features)
        T = np.random.RandomState(43).randn(3, n_features)
        P = lst_kernel(S, T)
        assert P.shape == (n_features, n_features)

    def test_identity_transform(self):
        """If source equals target, transform should be near identity."""
        rng = np.random.RandomState(42)
        S = rng.randn(20, 10)
        P = lst_kernel(S, S)
        # P @ S should approximate S
        reconstructed = P @ S
        # Not exact identity but should reconstruct well for square case
        assert np.allclose(reconstructed, S, atol=0.1)


# ============================================================================
# Tests for LST class
# ============================================================================

class TestLST:
    """Tests for the LST transformer."""

    def test_fit_returns_self(self):
        rng = np.random.RandomState(42)
        n_ch, n_samples = 8, 250
        X_source = rng.randn(10, n_ch, n_samples)
        X_target = rng.randn(5, n_ch, n_samples)
        y_source = np.array([0] * 5 + [1] * 5)
        y_target = np.array([0] * 3 + [1] * 2)

        lst = LST()
        result = lst.fit(X_source, y_source, X_target=X_target, y_target=y_target)
        assert result is lst

    def test_transform_shape(self):
        rng = np.random.RandomState(42)
        n_ch, n_samples = 8, 250
        X_source = rng.randn(10, n_ch, n_samples)
        X_target = rng.randn(5, n_ch, n_samples)
        y_source = np.array([0] * 5 + [1] * 5)
        y_target = np.array([0] * 3 + [1] * 2)

        lst = LST()
        lst.fit(X_source, y_source, X_target=X_target, y_target=y_target)
        X_transformed = lst.transform(X_source)
        assert X_transformed.shape == X_source.shape

    def test_transform_finite(self):
        rng = np.random.RandomState(42)
        n_ch, n_samples = 8, 250
        X_source = rng.randn(10, n_ch, n_samples)
        X_target = rng.randn(5, n_ch, n_samples)
        y_source = np.array([0] * 5 + [1] * 5)
        y_target = np.array([0] * 3 + [1] * 2)

        lst = LST()
        lst.fit(X_source, y_source, X_target=X_target, y_target=y_target)
        X_transformed = lst.transform(X_source)
        assert np.all(np.isfinite(X_transformed))
