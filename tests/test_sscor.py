# -*- coding: utf-8 -*-
# [METABCI-PYTEST-SUITE] Tests for SSCOR algorithms
#
# Tests for: metabci.brainda.algorithms.decomposition.sscor

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from metabci.brainda.algorithms.decomposition.sscor import (
    sscor_kernel,
    SSCOR,
    FBSSCOR,
)
from metabci.brainda.algorithms.decomposition.base import generate_filterbank


class TestSscorKernel:
    """Tests for the SSCOR kernel function."""

    def test_output_shapes(self, ssvep_data):
        X, y = ssvep_data
        # SSCOR kernel operates on single-class trials
        X_class0 = X[y == 0]
        W, D, A = sscor_kernel(X_class0)
        n_channels = X_class0.shape[1]
        assert W.shape == (n_channels, n_channels)
        assert D.shape == (n_channels,)
        assert A.shape == (n_channels, n_channels)

    def test_eigenvalues_sorted(self, ssvep_data):
        X, y = ssvep_data
        X_class0 = X[y == 0]
        _, D, _ = sscor_kernel(X_class0)
        # Eigenvalues should be sorted in descending order
        assert np.all(D[:-1] >= D[1:] - 1e-10)


class TestSSCOR:
    """Tests for the SSCOR estimator."""

    def test_fit_predict(self, ssvep_data):
        X, y = ssvep_data
        sscor = SSCOR(n_components=1)
        sscor.fit(X, y)
        labels = sscor.predict(X)
        assert labels.shape == y.shape

    def test_transform_shape(self, ssvep_data):
        X, y = ssvep_data
        sscor = SSCOR(n_components=1)
        sscor.fit(X, y)
        rhos = sscor.transform(X)
        n_classes = len(np.unique(y))
        assert rhos.shape == (X.shape[0], n_classes)


class TestFBSSCOR:
    """Tests for filter bank SSCOR."""

    def test_fit_predict(self, ssvep_data):
        X, y = ssvep_data
        wp = [(5, 90), (14, 90), (22, 90)]
        ws = [(3, 92), (12, 92), (20, 92)]
        fb = generate_filterbank(wp, ws, srate=250, order=4, rp=0.5)
        fw = np.array([(i + 1) ** (-1.25) + 0.25 for i in range(len(fb))])
        fbsscor = FBSSCOR(filterbank=fb, n_components=1, filterweights=fw)
        fbsscor.fit(X, y)
        labels = fbsscor.predict(X)
        assert labels.shape == y.shape
