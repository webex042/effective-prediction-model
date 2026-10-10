"""
Unit Tests for Stage 3: Neural Networks from Scratch & PyTorch Parity
=====================================================================

Verifies:
  1. Forward and backward passes for Linear, ReLU, Sigmoid, Tanh
  2. Numerical stability and gradients of SoftmaxCrossEntropyLoss
  3. Full backpropagation verification via two-sided numerical gradient checking
  4. Non-linear classification learning capability (XOR / non-linear clusters)
  5. Exact numerical parity between NumPy scratch MLP and PyTorch MLP
"""

import numpy as np
import pytest

from src.neural_networks.layers import (
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    SoftmaxCrossEntropyLoss,
    MSELoss,
)
from src.neural_networks.models_scratch import NeuralNetworkScratch
from src.neural_networks.models_pytorch import (
    PyTorchMLP,
    HAS_TORCH,
)


# =============================================================================
# 1. Layer Unit Tests
# =============================================================================

class TestLayers:
    def test_linear_forward_and_backward_shapes(self):
        N, D_in, D_out = 8, 4, 3
        X = np.random.randn(N, D_in)
        layer = Linear(D_in, D_out)

        out = layer.forward(X, training=True)
        assert out.shape == (N, D_out)

        grad_out = np.random.randn(N, D_out)
        grad_in = layer.backward(grad_out)

        assert grad_in.shape == (N, D_in)
        assert layer.dW.shape == (D_in, D_out)
        assert layer.db.shape == (D_out,)

    def test_relu_forward_and_backward(self):
        X = np.array([[-2.0, 0.0, 3.0], [5.0, -1.0, 0.5]])
        relu = ReLU()
        out = relu.forward(X, training=True)
        expected = np.array([[0.0, 0.0, 3.0], [5.0, 0.0, 0.5]])
        np.testing.assert_allclose(out, expected)

        grad_out = np.ones_like(X)
        grad_in = relu.backward(grad_out)
        expected_grad = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
        np.testing.assert_allclose(grad_in, expected_grad)

    def test_sigmoid_and_tanh(self):
        sig = Sigmoid()
        tanh = Tanh()
        X = np.array([-2.0, 0.0, 2.0])

        out_sig = sig.forward(X)
        assert out_sig[1] == 0.5
        assert 0.0 < out_sig[0] < 0.5 < out_sig[2] < 1.0

        out_tanh = tanh.forward(X)
        assert out_tanh[1] == 0.0
        assert -1.0 < out_tanh[0] < 0.0 < out_tanh[2] < 1.0


# =============================================================================
# 2. Loss Function Tests
# =============================================================================

class TestLosses:
    def test_softmax_cross_entropy_uniform_prior(self):
        loss_fn = SoftmaxCrossEntropyLoss()
        # 2 classes with equal logits -> loss should be ln(2) ≈ 0.6931
        logits = np.zeros((4, 2))
        y = np.array([0, 1, 0, 1])
        loss = loss_fn.forward(logits, y)
        np.testing.assert_allclose(loss, np.log(2.0), atol=1e-5)

    def test_softmax_cross_entropy_gradient(self):
        loss_fn = SoftmaxCrossEntropyLoss()
        logits = np.array([[2.0, 1.0], [0.5, 3.0]])
        y = np.array([0, 1])

        _ = loss_fn.forward(logits, y)
        grad = loss_fn.backward()

        assert grad.shape == logits.shape
        # Sum of softmax gradients across classes for each sample must be zero
        # because sum_c (p_c - y_c) = 1 - 1 = 0
        np.testing.assert_allclose(np.sum(grad, axis=1), 0.0, atol=1e-7)

    def test_mse_loss(self):
        loss_fn = MSELoss()
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.0, 2.0, 5.0])
        loss = loss_fn.forward(y_pred, y_true)
        # diff = [0, 0, -2], diff^2 = [0, 0, 4], mean = 4/3, 0.5 * 4/3 = 2/3
        np.testing.assert_allclose(loss, 2.0 / 3.0, atol=1e-5)


# =============================================================================
# 3. Numerical Gradient Checking (Backpropagation Verification)
# =============================================================================

class TestGradientCheck:
    def test_backprop_analytical_gradient_check(self):
        """
        Verify analytical gradients against two-sided numerical gradients:
        dJ/dθ ≈ (J(θ + ε) - J(θ - ε)) / (2ε).
        Requires relative error < 1e-5.
        """
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(8, 3))
        y = rng.choice([0, 1], size=8)

        model = NeuralNetworkScratch(in_features=3, hidden_dim=4, num_classes=2)
        check_result = model.gradient_check(X, y, epsilon=1e-5)

        assert check_result["passed"] is True
        assert check_result["max_relative_error"] < 1e-5


# =============================================================================
# 4. Learning Non-Linear Patterns
# =============================================================================

class TestModelTraining:
    def test_model_learns_non_linear_xor_problem(self):
        """A linear model cannot solve XOR; a 2-layer MLP with ReLU can."""
        np.random.seed(42)
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 25)
        y = np.array([0, 1, 1, 0] * 25)

        model = NeuralNetworkScratch(in_features=2, hidden_dim=16, num_classes=2)
        model.fit(X, y, epochs=150, batch_size=16, learning_rate=0.03, optimizer="adam")

        assert model.train_loss_history[-1] < model.train_loss_history[0]
        acc = model.score(X, y)
        assert acc >= 0.95


# =============================================================================
# 5. PyTorch Mathematical Parity
# =============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestPyTorchParity:
    def test_exact_logit_parity_with_copied_weights(self):
        """NumPy scratch MLP and PyTorch MLP must compute identical forward logits."""
        import torch

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(16, 5)).astype(np.float32)

        scratch = NeuralNetworkScratch(in_features=5, hidden_dim=12, num_classes=3)
        pytorch_model = PyTorchMLP(in_features=5, hidden_dim=12, num_classes=3)

        # Copy weights from NumPy to PyTorch
        pytorch_model.copy_weights_from_scratch(scratch)

        # Forward in NumPy
        logits_scratch = scratch.forward(X, training=False)

        # Forward in PyTorch
        with torch.no_grad():
            logits_pt = pytorch_model(torch.from_numpy(X)).numpy()

        # Check maximum absolute difference
        max_diff = np.max(np.abs(logits_scratch - logits_pt))
        assert max_diff < 1e-5, f"Logit difference too large: {max_diff}"

        # Check probability parity
        probs_scratch = scratch.predict_proba(X)
        probs_pt = pytorch_model.predict_proba(X)
        np.testing.assert_allclose(probs_scratch, probs_pt, atol=1e-5)

        # Check prediction match
        preds_scratch = scratch.predict(X)
        preds_pt = pytorch_model.predict(X)
        np.testing.assert_array_equal(preds_scratch, preds_pt)
