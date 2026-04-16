# -*- coding: utf-8 -*-
# [METABCI-PYTEST-SUITE] Tests for dynamic stopping algorithms
#
# Tests for: metabci.brainda.algorithms.dynamic_stopping

import pytest
import numpy as np

from metabci.brainda.algorithms.dynamic_stopping.bayes import (
    DummyKDE,
    Bayes,
)


# ============================================================================
# Tests for DummyKDE
# ============================================================================

class TestDummyKDE:
    """Tests for the DummyKDE helper class."""

    def test_constant_output(self):
        kde = DummyKDE(constant=0)
        result = kde([1, 2, 3])
        assert len(result) == 3
        assert all(r == 0 for r in result)

    def test_single_input(self):
        kde = DummyKDE(constant=1)
        result = kde([5.0])
        assert len(result) == 1


# ============================================================================
# Tests for Bayes dynamic stopping
# ============================================================================

class TestBayes:
    """Tests for the Bayes dynamic stopping algorithm."""

    def test_instantiation(self):
        """Test that Bayes can be instantiated with a dummy decoder."""
        from sklearn.base import BaseEstimator, ClassifierMixin

        class DummyDecoder(BaseEstimator, ClassifierMixin):
            def fit(self, X, y):
                self.classes_ = np.unique(y)
                return self

            def predict(self, X):
                return np.zeros(X.shape[0], dtype=int)

            def transform(self, X):
                return np.random.randn(X.shape[0], 2)

        decoder = DummyDecoder()
        bayes = Bayes(decoder=decoder)
        assert bayes is not None
        assert hasattr(bayes, 'decoder')
