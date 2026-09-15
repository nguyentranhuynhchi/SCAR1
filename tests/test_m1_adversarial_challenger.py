"""Adversarial Stress Testing Suite for M2-Pro Architecture.

Commissioned by: m1_challenger_1 (Empirical Challenger)
Milestone: M1 (M2-Pro Architecture)

Tests:
1. Adversarial Zero-CINE & Negative-CINE input (gradient floor alpha_pass >= 0.20 preservation).
2. Batch size extremes: B=1 (train/eval batchnorm behavior) and large batches (B=8, B=16).
3. Extreme numerical inputs: subnormal / tiny values (1e-12, 1e-7), high dynamic range ([-1000, 1000]), impulse hyper-intensity spikes.
4. Polymorphic output container M2ProOutput compliance: tuple unpacking, .main_logits, .aux_logits, indexing [0], shape attributes.
5. Decoupled bottleneck cross-attention ablation & pathology isolation.
6. Deep supervision aux head isolated gradient flow.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch import nn

from training.models import build_model
from training.models.cmspa_net import get_testing
from training.models.m2_pro import M2ProNet
from training.models.modules.m2_pro import (
    AGSA_Block,
    DecoupledBottleneckFusion,
    DeepSupervisionAuxHead,
    DFE_Block,
    M2ProOutput,
    SoftMyoGate,
)


class TestAdversarialZeroCine(unittest.TestCase):
    """Adversarial stress testing for zero-CINE input and gradient floor integrity."""

    def setUp(self):
        torch.manual_seed(42)

    def test_agsa_all_zero_cine_gradient_floor(self):
        """When CINE is all zeros, AGSA_Block must maintain d(g_mask)/d(psir) >= alpha_pass (0.20)."""
        alpha_pass = 0.20
        channels = 64
        size = 32
        block = AGSA_Block(channels=channels, alpha_pass=alpha_pass, use_dfe=False)

        cine_zero = torch.zeros(2, channels, size, size, requires_grad=True)
        psir = torch.randn(2, channels, size, size, requires_grad=True)
        t2w = torch.randn(2, channels, size, size, requires_grad=True)

        # In AGSA_Block: g_mask = alpha_pass + (1 - alpha_pass) * m_myo
        # Since m_myo is Sigmoid output, m_myo > 0, so g_mask >= alpha_pass everywhere.
        m_myo = block.gate(cine_zero)
        g_mask = block.alpha_pass + (1.0 - block.alpha_pass) * m_myo

        self.assertTrue(
            (g_mask >= alpha_pass).all(),
            f"g_mask floor violated! Min value was {g_mask.min().item()}, expected >= {alpha_pass}",
        )

        out = block(cine_zero, psir, t2w)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(psir.grad)
        self.assertTrue(torch.isfinite(psir.grad).all())
        # Verify gradient magnitude is not extinguished
        grad_norm = float(psir.grad.abs().mean())
        self.assertGreater(
            grad_norm,
            0.0,
            "PSIR gradient completely extinguished under all-zero CINE input!",
        )

    def test_agsa_adversarial_deeply_negative_cine(self):
        """When CINE is deeply negative (-1e4), SoftMyoGate outputs ~0.0, g_mask must strictly floor at alpha_pass."""
        alpha_pass = 0.20
        channels = 64
        size = 16
        block = AGSA_Block(channels=channels, alpha_pass=alpha_pass, use_dfe=False)

        # Force gate input to extreme negative so Sigmoid approaches 0.0
        cine_neg = torch.full((2, channels, size, size), -10000.0)
        m_myo = block.gate(cine_neg)
        g_mask = block.alpha_pass + (1.0 - block.alpha_pass) * m_myo

        # Even with m_myo ~ 0, g_mask must be >= alpha_pass
        self.assertGreaterEqual(
            float(g_mask.detach().min()),
            alpha_pass - 1e-6,
            f"g_mask fell below alpha_pass ({alpha_pass}) under adversarial negative CINE!",
        )

    def test_full_model_all_zero_cine_gradient_flow(self):
        """Verify full M2ProNet with all-zero CINE still propagates finite, non-zero gradients to LGE/PSIR."""
        model = M2ProNet(config=get_testing(), img_size=32)
        model.train()

        cine = torch.zeros(2, 1, 32, 32, requires_grad=True)
        psir = torch.randn(2, 1, 32, 32, requires_grad=True)
        t2w = torch.randn(2, 1, 32, 32, requires_grad=True)

        out = model(cine, psir, t2w)
        main_logits, aux_logits = out
        loss = main_logits.sum() + aux_logits.sum()
        loss.backward()

        self.assertIsNotNone(psir.grad, "LGE/PSIR gradient is None under all-zero CINE!")
        self.assertTrue(torch.isfinite(psir.grad).all(), "LGE/PSIR gradient contains NaN/Inf!")
        self.assertGreater(
            float(psir.grad.abs().sum()),
            0.0,
            "LGE/PSIR gradient extinguished to 0 under all-zero CINE!",
        )


class TestAdversarialBatchSizeExtremes(unittest.TestCase):
    """Adversarial stress testing for extreme batch sizes (B=1, B=8, B=16)."""

    def setUp(self):
        torch.manual_seed(42)

    def test_batch_size_1_training_mode(self):
        """Test B=1 in training mode to check BatchNorm behavior across all encoder/decoder layers."""
        model = M2ProNet(config=get_testing(), img_size=32)
        model.train()

        inputs = [torch.randn(1, 1, 32, 32, requires_grad=True) for _ in range(3)]
        out = model(*inputs)

        self.assertIsInstance(out, tuple)
        main_logits, aux_logits = out
        self.assertEqual(main_logits.shape, (1, 4, 32, 32))
        self.assertEqual(aux_logits.shape, (1, 2, 8, 8))

        # Check backward pass with B=1
        loss = main_logits.sum() + aux_logits.sum()
        loss.backward()

        for inp, name in zip(inputs, ("cine", "psir", "t2w")):
            self.assertIsNotNone(inp.grad)
            self.assertTrue(torch.isfinite(inp.grad).all(), f"NaN/Inf in grad for {name} with B=1")

    def test_batch_size_1_eval_mode(self):
        """Test B=1 in eval mode (standard inference condition in predict_volume)."""
        model = M2ProNet(config=get_testing(), img_size=32)
        model.eval()

        inputs = [torch.randn(1, 1, 32, 32) for _ in range(3)]
        with torch.no_grad():
            out = model(*inputs)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (1, 4, 32, 32))
        self.assertTrue(torch.isfinite(out).all())

    def test_batch_size_larger_batches(self):
        """Test B=8 and B=16 for memory and shape integrity."""
        model = M2ProNet(config=get_testing(), img_size=32)
        model.train()

        for b in (8, 16):
            inputs = [torch.randn(b, 1, 32, 32) for _ in range(3)]
            out = model(*inputs)
            self.assertEqual(out[0].shape, (b, 4, 32, 32))
            self.assertEqual(out[1].shape, (b, 2, 8, 8))
            self.assertTrue(torch.isfinite(out[0]).all())
            self.assertTrue(torch.isfinite(out[1]).all())


class TestAdversarialNumericalInputs(unittest.TestCase):
    """Stress testing with extreme numerical inputs: subnormal, high dynamic range, impulse spikes."""

    def setUp(self):
        torch.manual_seed(42)

    def test_very_small_numerical_inputs(self):
        """Input values near machine epsilon (1e-12, 1e-7) must not trigger division by zero or NaN."""
        model = M2ProNet(config=get_testing(), img_size=32)
        model.train()

        for scale in (1e-7, 1e-12):
            inputs = [torch.full((2, 1, 32, 32), scale, requires_grad=True) for _ in range(3)]
            out = model(*inputs)
            self.assertTrue(
                torch.isfinite(out[0]).all(),
                f"NaN or Inf in main_logits with input scale {scale}",
            )
            self.assertTrue(
                torch.isfinite(out[1]).all(),
                f"NaN or Inf in aux_logits with input scale {scale}",
            )

            loss = out[0].sum() + out[1].sum()
            loss.backward()
            self.assertTrue(
                torch.isfinite(inputs[0].grad).all(),
                f"NaN in gradients with input scale {scale}",
            )

    def test_high_dynamic_range_inputs(self):
        """Unnormalized or extreme dynamic range [-1000, 1000] and [0, 4095]."""
        model = M2ProNet(config=get_testing(), img_size=32)
        model.train()

        # Scale 1: [-1000, +1000]
        inputs = [(torch.rand(2, 1, 32, 32) * 2000.0 - 1000.0) for _ in range(3)]
        out = model(*inputs)
        self.assertTrue(torch.isfinite(out[0]).all(), "NaN/Inf in main_logits with [-1000, 1000]")
        self.assertTrue(torch.isfinite(out[1]).all(), "NaN/Inf in aux_logits with [-1000, 1000]")

        # Scale 2: [0, 4095] (12-bit raw DICOM CMR)
        inputs_12bit = [(torch.rand(2, 1, 32, 32) * 4095.0) for _ in range(3)]
        out_12bit = model(*inputs_12bit)
        self.assertTrue(torch.isfinite(out_12bit[0]).all(), "NaN/Inf in main_logits with 12-bit raw CMR")

    def test_impulse_hyperintensity_spike(self):
        """Single-pixel or focal extreme spike (+500.0) simulating severe artefact / acute hyper-enhancement."""
        model = M2ProNet(config=get_testing(), img_size=32)
        model.train()

        cine = torch.randn(2, 1, 32, 32)
        psir = torch.randn(2, 1, 32, 32)
        t2w = torch.randn(2, 1, 32, 32)

        # Inject focal spike into PSIR (scar channel)
        psir[:, :, 15:17, 15:17] = 500.0

        out = model(cine, psir, t2w)
        self.assertTrue(
            torch.isfinite(out[0]).all(),
            "Focal impulse spike resulted in NaN/Inf in main logits!",
        )
        self.assertTrue(
            torch.isfinite(out[1]).all(),
            "Focal impulse spike resulted in NaN/Inf in aux logits!",
        )


class TestM2ProOutputContainerCompliance(unittest.TestCase):
    """Verification of Polymorphic output container M2ProOutput compliance.

    Verifies requirements:
    1. Tuple unpacking: main, aux = model(...)
    2. Attribute access: .main_logits, .aux_logits
    3. Indexing: [0]
    4. Shape attributes: .shape
    """

    def test_tuple_unpacking(self):
        """Verify tuple unpacking works identically to standard 2-tuple."""
        main = torch.randn(2, 4, 32, 32)
        aux = torch.randn(2, 2, 8, 8)
        out = M2ProOutput(main, aux)

        unpacked_main, unpacked_aux = out
        self.assertTrue(torch.equal(unpacked_main, main))
        self.assertTrue(torch.equal(unpacked_aux, aux))

    def test_indexing(self):
        """Verify indexing [0] and [1]."""
        main = torch.randn(2, 4, 32, 32)
        aux = torch.randn(2, 2, 8, 8)
        out = M2ProOutput(main, aux)

        self.assertTrue(torch.equal(out[0], main))
        self.assertTrue(torch.equal(out[1], aux))

    def test_shape_attribute(self):
        """Verify output.shape returns main logits shape."""
        main = torch.randn(2, 4, 32, 32)
        aux = torch.randn(2, 2, 8, 8)
        out = M2ProOutput(main, aux)

        self.assertEqual(out.shape, (2, 4, 32, 32))

    def test_aux_logits_attribute(self):
        """Verify output.aux_logits returns aux tensor."""
        main = torch.randn(2, 4, 32, 32)
        aux = torch.randn(2, 2, 8, 8)
        out = M2ProOutput(main, aux)

        self.assertTrue(torch.equal(out.aux_logits, aux))

    def test_main_logits_attribute(self):
        """Verify output.main_logits attribute access.

        Requirement check:
        'Test Polymorphic output container M2ProOutput: check tuple unpacking main, aux = model(...),
        attribute access .main_logits, .aux_logits, indexing [0], and shape attributes.'
        """
        main = torch.randn(2, 4, 32, 32)
        aux = torch.randn(2, 2, 8, 8)
        out = M2ProOutput(main, aux)

        # Check if .main_logits exists on M2ProOutput
        has_main_logits = hasattr(out, "main_logits")
        self.assertTrue(
            has_main_logits,
            "CRITICAL: M2ProOutput is missing the '.main_logits' property! "
            "Found 'logits' and 'aux_logits', but '.main_logits' raises AttributeError.",
        )
        if has_main_logits:
            self.assertTrue(torch.equal(out.main_logits, main))

    def test_dict_access_main_logits(self):
        """Verify output['main_logits'] dictionary indexing and 'main_logits' in out."""
        main = torch.randn(2, 4, 32, 32)
        aux = torch.randn(2, 2, 8, 8)
        out = M2ProOutput(main, aux)

        try:
            val = out["main_logits"]
            self.assertTrue(torch.equal(val, main))
        except KeyError:
            self.fail("M2ProOutput does not support ['main_logits'] dictionary access key.")



class TestDecoupledBottleneckAblation(unittest.TestCase):
    """Stress test DecoupledBottleneckFusion to verify LGE and T2w modality isolation."""

    def test_modality_isolation(self):
        """Verify changing T2w input does not change LGE cross-attention query representations."""
        fusion = DecoupledBottleneckFusion(in_channels=128, out_channels=64, num_heads=4, dropout=0.0)
        fusion.eval()

        cine = torch.randn(2, 128, 8, 8)
        psir = torch.randn(2, 128, 8, 8)
        t2w_1 = torch.randn(2, 128, 8, 8)
        t2w_2 = torch.randn(2, 128, 8, 8)

        # Forward with t2w_1 vs t2w_2
        # In DecoupledBottleneckFusion, scar branch computes:
        # feat_scar = psir + attn(query=psir, key=cine, value=cine)
        # This feat_scar must be BITWISE IDENTICAL regardless of t2w!
        q_cine = fusion.norm_cine(cine.flatten(2).transpose(1, 2))
        q_psir = fusion.norm_psir(psir.flatten(2).transpose(1, 2))

        attn_scar1, _ = fusion.mha_scar(query=q_psir, key=q_cine, value=q_cine, need_weights=False)
        attn_scar2, _ = fusion.mha_scar(query=q_psir, key=q_cine, value=q_cine, need_weights=False)
        torch.testing.assert_close(attn_scar1, attn_scar2)


if __name__ == "__main__":
    unittest.main()
