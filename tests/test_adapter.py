import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from vlm_probing.adapters import HookPoint, ModelReadout, TorchModelAdapter
from vlm_probing.core import ProbeResult
from vlm_probing.metrics import sequence_logprob


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 3, bias=False)
        self.head = nn.Linear(3, 2, bias=False)

    def forward(self, x):
        return self.head(self.hidden(x))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = SmallModel()
        self.adapter = TorchModelAdapter(self.model, {"hidden": HookPoint("hidden")})

    def test_noop_and_actual_patch_effect(self):
        x = torch.ones(1, 3)
        original = self.model(x).detach()
        traced = self.adapter.run({"x": x}, capture=["hidden"])
        torch.testing.assert_close(traced.logits, original)
        no_op = self.adapter.run({"x": x}, interventions={"hidden": lambda h: h})
        torch.testing.assert_close(no_op.logits, original)
        patched = self.adapter.run({"x": x}, interventions={"hidden": torch.zeros_like})
        torch.testing.assert_close(patched.logits, torch.zeros_like(original))
        self.assertFalse(torch.allclose(original, patched.logits))

    def test_exception_cleans_hooks_and_restores_mixed_modes(self):
        self.model.train()
        self.model.head.eval()
        def fail(value):
            raise RuntimeError("deliberate")
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            self.adapter.run({"x": torch.ones(1, 3)}, interventions={"hidden": fail})
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.head.training)
        self.assertEqual(len(self.model.hidden._forward_hooks), 0)
        self.adapter.run({"x": torch.ones(1, 3)})

    def test_gradient_capture_on_actual_output_path(self):
        trace = self.adapter.run({"x": torch.ones(1, 3)}, capture=["hidden"], grad=True)
        gradient, = torch.autograd.grad(trace.logits.sum(), trace.activations["hidden"])
        torch.testing.assert_close(gradient, self.model.head.weight.sum(0)[None])

    def test_positional_input_edit(self):
        adapter = TorchModelAdapter(self.model, {"head_in": HookPoint("head", "input", 0)})
        result = adapter.run({"x": torch.ones(1, 3)}, interventions={"head_in": torch.zeros_like})
        torch.testing.assert_close(result.logits, torch.zeros(1, 2))

    def test_downstream_inplace_gradient_capture_rejected(self):
        class Inplace(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(2, 2)
            def forward(self, x):
                hidden = self.layer(x)
                hidden.mul_(2)
                return hidden.sum()
        model = Inplace()
        adapter = TorchModelAdapter(model, {"hidden": HookPoint("layer")})
        with self.assertRaisesRegex(RuntimeError, "mutated in-place"):
            adapter.run({"x": torch.ones(1, 2)}, capture=["hidden"], grad=True)
        self.assertEqual(len(model.layer._forward_hooks), 0)

    def test_missing_and_repeated_sites_raise(self):
        class Repeated(nn.Module):
            def __init__(self):
                super().__init__()
                self.site = nn.Identity()
                self.unused = nn.Identity()
            def forward(self, x):
                return self.site(self.site(x))
        model = Repeated()
        adapter = TorchModelAdapter(model, {"site": HookPoint("site"), "unused": HookPoint("unused")})
        with self.assertRaisesRegex(RuntimeError, "more than once"):
            adapter.run({"x": torch.ones(1)}, capture=["site"])
        with self.assertRaisesRegex(RuntimeError, "not executed"):
            adapter.run({"x": torch.ones(1)}, capture=["unused"])
        self.assertEqual(len(model.site._forward_hooks), 0)

    def test_readout_normalizes_exactly_once(self):
        norm = nn.LayerNorm(3)
        head = nn.Linear(3, 2)
        x = torch.randn(2, 3)
        raw_readout = ModelReadout(head, norm=norm)
        normalized_readout = ModelReadout(head, norm=norm, input_normalized=True)
        torch.testing.assert_close(raw_readout(x), normalized_readout(norm(x)))

    def test_answer_shift_and_span(self):
        logits = torch.tensor([[[0., 1., 2.], [2., 0., 0.], [9., 9., 9.]]])
        ids = torch.tensor([[1, 2, 0]])
        answer = torch.tensor([[False, True, True]])
        expected = logits[0, 0].log_softmax(-1)[2] + logits[0, 1].log_softmax(-1)[0]
        torch.testing.assert_close(sequence_logprob(logits, ids, answer), expected[None])
        torch.testing.assert_close(sequence_logprob(logits, ids, answer, reduction="mean"), expected[None] / 2)

    def test_result_save_detaches_tensors(self):
        with tempfile.TemporaryDirectory() as folder:
            ProbeResult("test", {"scores": torch.ones(2, requires_grad=True)}, {"axes": ["sample"]}).save(folder)
            saved = torch.load(Path(folder) / "tensors.pt", weights_only=True)
            self.assertFalse(saved["scores"].requires_grad)
            self.assertTrue((Path(folder) / "metadata.json").exists())

    def test_masked_prompt_logits_do_not_pollute_score_gradients(self):
        logits = torch.tensor([[[-torch.inf, -torch.inf], [0., 1.], [0., 0.]]], requires_grad=True)
        score = sequence_logprob(logits, torch.tensor([[0, 0, 1]]),
                                 torch.tensor([[False, False, True]]))
        gradient, = torch.autograd.grad(score.sum(), logits)
        self.assertTrue(torch.isfinite(gradient).all())
        torch.testing.assert_close(gradient[:, 0], torch.zeros(1, 2))


if __name__ == "__main__":
    unittest.main()
