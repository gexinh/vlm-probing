"""Public head initialization agrees with a raw vocabulary projection."""
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from tiny_model import TinyModel
from vlm_probing import Prober


class AttentionLensInitializationTests(unittest.TestCase):
    def test_public_initialization_controls_readout_and_initial_loss(self):
        torch.manual_seed(47)
        model = TinyModel()
        probe = Prober(model)
        inputs = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                  "image_tokens": torch.randn(2, 2, 6)}
        before = {name: parameter.detach().clone()
                  for name, parameter in model.named_parameters()}
        # A nonzero bias catches both transposition and bias forwarding mistakes.
        weights = model.lm_head.weight.detach().clone()
        bias = torch.linspace(0.1, 0.7, weights.shape[0])
        trace = probe.adapter.run(inputs, capture=[probe.spec.heads[0]])
        heads = probe._heads(0, trace.activations[probe.spec.heads[0]])[:, -1]
        expected = F.linear(heads, weights, bias)
        teacher_log_probs = trace.logits[:, -1].log_softmax(-1)
        expected_loss = F.kl_div(expected.sum(-2).log_softmax(-1), teacher_log_probs,
                                 reduction="none", log_target=True).sum(-1).mean()

        method = probe.lens.attention(layers=[0])
        # Optimization still executes. Its tiny step leaves the initialized
        # projection unchanged within floating-point tolerance for this check.
        method.fit(inputs, initial_unembedding=weights, initial_bias=bias,
                   steps=1, lr=1e-10)
        result = method.run(inputs)
        torch.testing.assert_close(result.tensors["head_logits"][0], expected)
        self.assertAlmostEqual(method.losses[0][0], expected_loss.item(), places=6)
        # Individual heads use the raw learned map, without final model norm.
        self.assertFalse(torch.allclose(expected, F.linear(model.norm(heads), weights, bias)))
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name])
            self.assertIsNone(parameter.grad)

        fitted = method.kernels[0].decoders.weight.detach().clone()
        with self.assertRaisesRegex(ValueError, "requires new Attention Lens"):
            method.fit(inputs, initial_unembedding=weights, steps=1)
        torch.testing.assert_close(method.kernels[0].decoders.weight, fitted)
        with self.assertRaisesRegex(ValueError, "only supported by Attention Lens"):
            probe.lens.tuned(layers=[0]).fit(inputs, initial_unembedding=weights, steps=1)


if __name__ == "__main__":
    unittest.main()
