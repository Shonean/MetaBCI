# -*- coding: utf-8 -*-
# [METABCI-PYTEST-SUITE] Tests for deep learning models
#
# Tests for: metabci.brainda.algorithms.deep_learning

import pytest
import numpy as np
import torch

from metabci.brainda.algorithms.deep_learning.eegnet import EEGNet
from metabci.brainda.algorithms.deep_learning.shallownet import ShallowNet
from metabci.brainda.algorithms.deep_learning.deepnet import Deep4Net


# ============================================================================
# Constants
# ============================================================================

N_CHANNELS = 8
N_SAMPLES = 250
N_CLASSES = 2
BATCH_SIZE = 4


# ============================================================================
# Tests for EEGNet
# ============================================================================

class TestEEGNet:
    """Tests for the EEGNet architecture."""

    def test_instantiation(self):
        """Test that EEGNet can be instantiated."""
        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        assert model is not None

    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        # Access the underlying PyTorch module
        net = model.module if hasattr(model, 'module') else model
        x = torch.randn(BATCH_SIZE, 1, N_CHANNELS, N_SAMPLES)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (BATCH_SIZE, N_CLASSES)

    def test_output_finite(self):
        """Test that output contains no NaN/Inf."""
        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        net = model.module if hasattr(model, 'module') else model
        x = torch.randn(BATCH_SIZE, 1, N_CHANNELS, N_SAMPLES)
        with torch.no_grad():
            out = net(x)
        assert torch.all(torch.isfinite(out))

    def test_different_n_classes(self):
        """Test EEGNet with different number of classes."""
        for n_classes in [2, 4, 10]:
            model = EEGNet(
                n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=n_classes
            )
            net = model.module if hasattr(model, 'module') else model
            x = torch.randn(2, 1, N_CHANNELS, N_SAMPLES)
            with torch.no_grad():
                out = net(x)
            assert out.shape == (2, n_classes)


# ============================================================================
# Tests for ShallowNet
# ============================================================================

class TestShallowNet:
    """Tests for the ShallowNet architecture."""

    def test_instantiation(self):
        model = ShallowNet(
            n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES
        )
        assert model is not None

    def test_forward_pass_shape(self):
        model = ShallowNet(
            n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES
        )
        net = model.module if hasattr(model, 'module') else model
        x = torch.randn(BATCH_SIZE, 1, N_CHANNELS, N_SAMPLES)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (BATCH_SIZE, N_CLASSES)


# ============================================================================
# Tests for Deep4Net
# ============================================================================

class TestDeep4Net:
    """Tests for the Deep4Net architecture."""

    def test_instantiation(self):
        model = Deep4Net(
            n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES
        )
        assert model is not None

    def test_forward_pass_shape(self):
        model = Deep4Net(
            n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES
        )
        net = model.module if hasattr(model, 'module') else model
        x = torch.randn(BATCH_SIZE, 1, N_CHANNELS, N_SAMPLES)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (BATCH_SIZE, N_CLASSES)

    def test_output_finite(self):
        model = Deep4Net(
            n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES
        )
        net = model.module if hasattr(model, 'module') else model
        x = torch.randn(BATCH_SIZE, 1, N_CHANNELS, N_SAMPLES)
        with torch.no_grad():
            out = net(x)
        assert torch.all(torch.isfinite(out))


# ============================================================================
# Tests for sklearn-compatible interface (skorch wrapper)
# ============================================================================

class TestSkorchInterface:
    """Tests for the skorch NeuralNetClassifier wrapper."""

    def test_eegnet_has_fit_method(self):
        """EEGNet wrapped by skorch should have fit/predict methods."""
        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_eegnet_fit_predict(self):
        """Test that EEGNet can fit on small synthetic data and predict."""
        model = EEGNet(
            n_channels=N_CHANNELS,
            n_samples=N_SAMPLES,
            n_classes=N_CLASSES,
            max_epochs=2,
            batch_size=8,
            verbose=False,
        )
        # Create small training data
        rng = np.random.RandomState(42)
        X_train = rng.randn(20, 1, N_CHANNELS, N_SAMPLES).astype(np.float32)
        y_train = np.array([0] * 10 + [1] * 10).astype(np.int64)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        assert y_pred.shape == y_train.shape
        assert set(y_pred).issubset({0, 1})
