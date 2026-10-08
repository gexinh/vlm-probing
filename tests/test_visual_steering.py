"""Contrastive VSV extraction in real residual coordinates."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tiny_model import TinyModel
from vlm_probing import Prober
from vlm_probing.causal.visual_steering import VisualSteering


class VisualSteeringTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.probe = Prober(TinyModel())
        self.positive = {"input_ids": torch.tensor([[1, 3, 5]]),
                         "image_tokens": torch.randn(1, 2, 6)}
        self.negative = {"input_ids": torch.tensor([[1, 3, 5]]),
                         "image_tokens": torch.empty(1, 0, 6)}

    def test_vectors_are_context_ends_with_different_sequence_lengths(self):
        method = VisualSteering.from_inputs(self.probe, self.positive, self.negative)
        sites = [self.probe.spec.residuals[layer] for layer in [0, 1]]
        pos = self.probe._run(self.positive, capture=sites)
        neg = self.probe._run(self.negative, capture=sites)
        for layer, site in enumerate(sites):
            torch.testing.assert_close(method.directions[layer][:, 0],
                                       pos.activations[site][:, -1] - neg.activations[site][:, -1])
        self.assertEqual(method.metadata["positive_positions"], [[0, 4]])
        self.assertEqual(method.metadata["negative_positions"], [[0, 2]])
        self.assertTrue(self.probe.model.training)
        self.assertTrue(all(p.grad is None for p in self.probe.model.parameters()))

    def test_identity_context_produces_exact_zero_vectors(self):
        method = VisualSteering.from_inputs(self.probe, self.positive, self.positive)
        self.assertTrue(all(torch.count_nonzero(x) == 0 for x in method.directions.values()))

    def test_rejects_multitoken_or_nonfinite_directions(self):
        with self.assertRaisesRegex(ValueError, "one selected"):
            VisualSteering.from_inputs(self.probe, self.positive, self.negative, tokens="all")
        with self.assertRaisesRegex(ValueError, "finite"):
            VisualSteering(self.probe, {0: torch.full((1, 6), float("nan"))})

    def test_constructor_does_not_alias_caller_vectors(self):
        vector = torch.ones(1, 6)
        method = VisualSteering(self.probe, {0: vector})
        vector.zero_()
        torch.testing.assert_close(method.directions[0], torch.ones(1, 1, 6))

    def test_fixed_prefix_keeps_original_prediction_boundary(self):
        from demos.inputs import add_prefix
        prefixed = add_prefix(self.positive, [2, 4], probe=self.probe)
        layout = prefixed.layout
        self.assertEqual(layout.select("prediction").nonzero().tolist(), [[0, 4], [0, 5], [0, 6]])
        self.assertEqual(layout.select("last_prompt").nonzero().tolist(), [[0, 4]])
        self.assertEqual(layout.visual.nonzero().tolist(), [[0, 0], [0, 1]])
        self.assertEqual(layout.token_ids.shape, layout.valid.shape)


if __name__ == "__main__":
    unittest.main()
