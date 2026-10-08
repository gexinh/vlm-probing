"""CPU checks of interventions against analytic effects and a tiny attention model."""

import unittest

import torch

from vlm_probing.causal import (
    ActivationPatching, AttentionKnockout, AttributionPatching,
    BaseCausal, EAPIG, PathPatching, Steering,
)


class CausalTests(unittest.TestCase):
    def setUp(self):
        self.receiver = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        self.source = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
        self.mask = torch.tensor([[[True], [False]]])

    def test_family_is_abstract_and_concrete_methods_share_it(self):
        with self.assertRaises(TypeError):
            BaseCausal()
        for cls in (PathPatching, ActivationPatching, AttentionKnockout,
                    AttributionPatching, EAPIG, Steering):
            self.assertIsInstance(cls(), BaseCausal)

    def test_self_patch_and_actual_model_effect(self):
        # The replayed downstream computation depends only on the first token.
        head = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[2.0, -1.0]]))
        runner = lambda activation: head(activation[:, 0]).sum()
        same = ActivationPatching().run(self.receiver, self.receiver, runner=runner)
        self.assertEqual(same.tensors["effect"].item(), 0.0)
        result = ActivationPatching().run(
            self.receiver, self.source, mask=self.mask, runner=runner,
        )
        self.assertEqual(result.tensors["effect"].item(), 4.0)
        torch.testing.assert_close(result.tensors["edited"][:, 1], self.receiver[:, 1])
        self.assertTrue(result.metadata["effect_measured"])

    def test_no_input_mutation_and_tensor_only_result_is_explicit(self):
        original = self.receiver.clone()
        source = self.source.clone()
        result = ActivationPatching().run(self.receiver, self.source, mask=self.mask)
        torch.testing.assert_close(self.receiver, original)
        torch.testing.assert_close(self.source, source)
        self.assertFalse(result.metadata["effect_measured"])
        self.assertNotIn("effect", result.tensors)
        Steering().run(self.receiver, torch.tensor([1.0, -1.0]), mask=self.mask)
        torch.testing.assert_close(self.receiver, original)

    def test_patch_validates_shape_dtype_and_mask(self):
        with self.assertRaises(ValueError):
            ActivationPatching().run(self.receiver, self.source[:, :1])
        with self.assertRaises(ValueError):
            ActivationPatching().run(self.receiver, self.source.double())
        with self.assertRaises(TypeError):
            ActivationPatching().run(self.receiver, self.source, mask=torch.ones(1))
        with self.assertRaises(ValueError):
            ActivationPatching().run(self.receiver, self.source, mask=torch.ones(3, dtype=torch.bool))

    def test_attention_knockout_changes_value_aggregation_and_merges_masks(self):
        logits = torch.tensor([[0.0, 0.0, float("-inf")]])
        before = logits.clone()
        values = torch.tensor([[2.0], [8.0], [100.0]])
        result = AttentionKnockout().run(
            logits, blocked=torch.tensor([[True, False, False]]),
            runner=lambda scores: (scores.softmax(dim=-1) @ values).sum(),
        )
        self.assertEqual(result.tensors["effect"].item(), 3.0)
        torch.testing.assert_close(result.tensors["probabilities"], torch.tensor([[0.0, 1.0, 0.0]]))
        torch.testing.assert_close(logits, before)
        with self.assertRaises(ValueError):
            AttentionKnockout().run(logits, blocked=torch.tensor([[True, True, False]]))
        with self.assertRaises(ValueError):
            AttentionKnockout().run(torch.tensor([[float("inf"), 0.0]]), blocked=torch.tensor(False))

    def test_attribution_sign_and_gradient_use_receiver(self):
        receiver = torch.tensor([2.0, 3.0])
        source = torch.tensor([4.0, 1.0])
        result = AttributionPatching().run(receiver, source, metric=lambda x: x.square().sum())
        torch.testing.assert_close(result.tensors["gradient"], 2 * receiver)
        torch.testing.assert_close(result.tensors["attribution"], torch.tensor([8.0, -12.0]))
        self.assertEqual(result.tensors["estimated_effect"].item(), -4.0)
        self.assertIsNone(receiver.grad)
        with self.assertRaises(ValueError):
            AttributionPatching().run(receiver, source)

    def test_integrated_gradient_edges_match_analytic_quadratic(self):
        # Two actual input edges to a sum-of-squares downstream graph.
        receiver = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        source = torch.tensor([[2.0, 4.0], [1.0, 3.0]], dtype=torch.float64)
        before = receiver.clone()
        result = EAPIG().run(
            receiver, source, edge_names=["left->readout", "right->readout"],
            metric=lambda edges: edges.square().sum(), steps=8,
        )
        expected = source.square() - receiver.square()
        torch.testing.assert_close(result.tensors["attribution"], expected)
        torch.testing.assert_close(result.tensors["edge_scores"], expected.sum(dim=-1))
        self.assertLess(abs(result.tensors["completeness_error"].item()), 1e-12)
        torch.testing.assert_close(receiver, before)
        self.assertIsNone(receiver.grad)

    def test_integrated_gradient_mask_and_edge_contract(self):
        metric = lambda x: x.square().sum()
        receiver = torch.tensor([[1.0], [2.0]])
        source = torch.tensor([[3.0], [4.0]])
        result = EAPIG().run(receiver, source, edge_names=["a->z", "b->z"],
                            metric=metric, mask=torch.tensor([[True], [False]]), steps=4)
        torch.testing.assert_close(result.tensors["edge_scores"], torch.tensor([8.0, 0.0]))
        self.assertEqual(result.tensors["effect"].item(), 8.0)
        with self.assertRaises(ValueError):
            EAPIG().run(receiver, source, edge_names=["x", "x"], metric=metric)
        with self.assertRaises(ValueError):
            EAPIG().run(receiver, source, edge_names=["x", "y"], metric=metric, steps=0)

    def test_steering_norms_masks_and_cancelled_vector(self):
        original = self.receiver.clone()
        result = Steering().run(
            self.receiver, torch.tensor([0.5, -0.5]), mask=self.mask, preserve_norm=True,
        )
        torch.testing.assert_close(result.tensors["edited"].norm(dim=-1), self.receiver.norm(dim=-1))
        torch.testing.assert_close(result.tensors["edited"][:, 1], self.receiver[:, 1])
        torch.testing.assert_close(self.receiver, original)
        with self.assertRaises(ValueError):
            Steering().run(self.receiver, -self.receiver, preserve_norm=True)
        with self.assertRaises(ValueError):
            Steering().run(self.receiver, self.source, preserve_norm=True,
                           mask=torch.tensor([True, False]))


if __name__ == "__main__":
    unittest.main()
