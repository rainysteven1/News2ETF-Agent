"""Tests for signals models under ``trainer/src/models``."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from trainer.src.models.signals import SpatialDropout, TCNFanIn, export_tcn_fanin_to_onnx


class TestSpatialDropout:
    def test_training_mode_applies_dropout(self):
        """Training mode should zero some sub-dims and scale remaining values."""
        sd = SpatialDropout(p=0.3)
        sd.train()
        x = torch.ones(2, 3, 47, 6)
        out = sd(x)
        assert out.shape == x.shape
        # Inverted dropout: non-zeroed values are scaled up by 1/(1-p)
        # So non-zeroed values should be 1/(1-0.3) ≈ 1.4286
        # Check that not all values are 1.0 (dropout was applied)
        assert not torch.allclose(out, x), "SpatialDropout should modify values in training mode"

    def test_eval_mode_identity(self):
        """Eval mode should return input unchanged."""
        sd = SpatialDropout(p=0.3)
        sd.eval()
        x = torch.randn(2, 3, 47, 6)
        out = sd(x)
        assert torch.allclose(out, x)


class TestTCNFanIn:
    def test_output_shape(self):
        """Input (4,5,47,6) → reg (4,8), cls (4,1)."""
        model = TCNFanIn(n_sub=47, n_meta=8, input_size=6, hidden_size=32, num_layers=2)
        model.eval()
        x = torch.randn(4, 5, 47, 6)
        reg, cls = model(x)
        assert reg.shape == (4, 8), f"Expected (4,8), got {reg.shape}"
        assert cls.shape == (4, 1), f"Expected (4,1), got {cls.shape}"

    def test_output_range(self):
        """reg in [-1,1], cls in [0,1]."""
        model = TCNFanIn(n_sub=47, n_meta=8, input_size=6, hidden_size=32, num_layers=2)
        model.eval()
        x = torch.randn(8, 5, 47, 6)
        reg, cls = model(x)
        assert (reg >= -1).all() and (reg <= 1).all(), "reg out of [-1,1]"
        assert (cls >= 0).all() and (cls <= 1).all(), "cls out of [0,1]"

    def test_backward(self):
        """Gradient should flow through the model."""
        model = TCNFanIn(n_sub=47, n_meta=8, input_size=6, hidden_size=32, num_layers=2)
        x = torch.randn(2, 5, 47, 6, requires_grad=True)
        reg, cls = model(x)
        loss = reg.sum() + cls.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_learnable_scale(self):
        """model.scale should be a learnable Parameter."""
        model = TCNFanIn(n_sub=47, n_meta=8)
        assert isinstance(model.scale, nn.Parameter)
        assert model.scale.shape == (1,)

    def test_export_onnx(self):
        """Export to ONNX and verify file exists."""
        pytest.importorskip("onnx")
        model = TCNFanIn(n_sub=47, n_meta=8, input_size=6, hidden_size=32, num_layers=2)
        model.eval()
        with tempfile.TemporaryDirectory() as td:
            onnx_path = Path(td) / "tcn_fanin.onnx"
            export_tcn_fanin_to_onnx(model, onnx_path, seq_len=5, n_sub=47, input_size=6)
            assert onnx_path.exists()
            assert onnx_path.stat().st_size > 1000, "ONNX file too small"
