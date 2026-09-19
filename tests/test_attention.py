"""CPU semantic checks for attention algorithms, with no model downloads."""

import math
import unittest

import torch

from vlm_probing.attention import (
    AttentionProfile, AttentionRelevance, AttentionReweight, AttentionRollout,
    AttentionTemperature, BaseAttention, HeadLogitAttribution,
)


class AttentionTests(unittest.TestCase):
    def test_family_base_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseAttention()

    def test_profile_respects_each_examples_groups_and_padding(self):
        attention = torch.tensor([[[[0.25, 0.75, 0.0], [0.0, 0.0, 0.0]]],
                                  [[[0.0, 0.5, 0.5], [0.0, 1.0, 0.0]]]])
        qmask = torch.tensor([[True, False], [True, True]])
        kmask = torch.tensor([[True, True, False], [False, True, True]])
        groups = {"visual": torch.tensor([[True, False, False], [False, False, True]]),
                  "text": torch.tensor([[False, True, False], [False, True, False]])}
        result = AttentionProfile().run(attention, groups=groups,
                                        query_mask=qmask, key_mask=kmask)
        expected = torch.tensor([[[[0.25, 0.75], [0.0, 0.0]]],
                                 [[[0.5, 0.5], [0.0, 1.0]]]])
        torch.testing.assert_close(result.tensors["group_mass"], expected)
        self.assertAlmostEqual(result.tensors["entropy"][1, 0, 0].item(), math.log(2), places=6)
        self.assertEqual(result.tensors["entropy"][0, 0, 1].item(), 0)
        torch.testing.assert_close(result.tensors["valid_queries"], qmask)

    def test_profile_rejects_invalid_probabilities_and_modality_overlap(self):
        profile = AttentionProfile()
        for values in ([0.4, 0.4], [-0.1, 1.1], [float("nan"), 0.0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                profile.run(torch.tensor(values).reshape(1, 1, 1, 2))
        attention = torch.tensor([[[[0.5, 0.5]]]])
        group = torch.tensor([[True, False]])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            profile.run(attention, groups={"vision": group, "text": group})
        with self.assertRaisesRegex(ValueError, "invalid key"):
            profile.run(attention, key_mask=group)

    def test_head_attribution_is_additive_with_full_residual_layer_norm_scale(self):
        generator = torch.Generator().manual_seed(8)
        heads = torch.randn(2, 3, 4, 5, generator=generator)
        residual = heads.sum(1)
        weight = torch.randn(7, 5, generator=generator)
        gain = torch.randn(5, generator=generator)
        scale = torch.rsqrt(residual.var(-1, unbiased=False, keepdim=True) + 1e-5)
        result = HeadLogitAttribution().run(heads, weight * gain, fixed_scale=scale, center=True)
        expected = torch.nn.functional.layer_norm(residual, (5,), weight=gain) @ weight.T
        torch.testing.assert_close(result.tensors["head_logits"].sum(1), expected,
                                   rtol=1e-5, atol=2e-6)
        # Independent per-head normalization is not an additive decomposition.
        independent = torch.nn.functional.layer_norm(heads, (5,), weight=gain).sum(1) @ weight.T
        self.assertFalse(torch.allclose(independent, expected))
        with self.assertRaisesRegex(ValueError, "shared across heads"):
            HeadLogitAttribution().run(heads, weight, fixed_scale=torch.ones(2, 3, 4, 1))

    def test_rollout_follows_layer_order_and_residual_paths(self):
        # Layer 0 maps both positions to source 0; layer 1 swaps positions.
        first = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        second = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        stack = torch.stack([first, second])[:, None, None]
        pure = AttentionRollout(residual=False).run(stack).tensors["rollout"]
        torch.testing.assert_close(pure, first[None])
        residual = AttentionRollout().run(stack).tensors["rollout"]
        torch.testing.assert_close(residual, torch.tensor([[[0.75, 0.25], [0.75, 0.25]]]))

    def test_rollout_padding_and_empty_min_reduction(self):
        matrix = torch.tensor([[1.0, 0.0, 0.0], [0.25, 0.75, 0.0], [0.0, 0.0, 0.0]])
        mask = torch.tensor([[True, True, False]])
        result = AttentionRollout().run(matrix[None, None, None], valid_tokens=mask)
        torch.testing.assert_close(result.tensors["rollout"].sum(-1), mask.float())
        heads = torch.stack([torch.eye(2), torch.eye(2).flip(-1)])[None, None]
        with self.assertRaisesRegex(ValueError, "empty valid row"):
            AttentionRollout(head_reduction="min", residual=False).run(heads)

    def test_relevance_uses_target_gradient_and_is_not_normalized(self):
        attention = torch.tensor([[[[[0.8, 0.2], [0.3, 0.7]]]]], requires_grad=True)
        # An explicit target prefers key 0 and penalizes key 1 at query 1.
        score = 2 * attention[0, 0, 0, 1, 0] - attention[0, 0, 0, 1, 1]
        gradients, = torch.autograd.grad(score, attention)
        result = AttentionRelevance().run(attention, gradients)
        torch.testing.assert_close(result.tensors["relevance"],
                                   torch.tensor([[[1.0, 0.0], [0.6, 1.0]]]))
        self.assertFalse(result.metadata["row_normalized"])
        zero = AttentionRelevance().run(attention, torch.zeros_like(gradients))
        torch.testing.assert_close(zero.tensors["relevance"], torch.eye(2)[None])
        with self.assertRaisesRegex(ValueError, "finite"):
            AttentionRelevance().run(attention, torch.full_like(gradients, float("nan")))

    def test_reweight_distinguishes_mass_scaling_and_redistribution(self):
        attention = torch.tensor([[[[0.25, 0.75]]]])
        weight = torch.tensor([2.0, 1.0])
        normalized = AttentionReweight().run(attention, weight).tensors["attention"]
        unnormalized = AttentionReweight(renormalize=False).run(attention, weight).tensors["attention"]
        torch.testing.assert_close(normalized, torch.tensor([[[[0.4, 0.6]]]]))
        torch.testing.assert_close(unnormalized, torch.tensor([[[[0.5, 0.75]]]]))
        values = torch.tensor([[[[10.0], [0.0]]]])
        self.assertAlmostEqual((normalized @ values).item(), 4.0)
        self.assertAlmostEqual((unnormalized @ values).item(), 5.0)
        torch.testing.assert_close(attention, torch.tensor([[[[0.25, 0.75]]]]))
        for renormalize in (True, False):
            with self.assertRaisesRegex(ValueError, "every key"):
                AttentionReweight(renormalize=renormalize).run(attention, 0.0)

    def test_temperature_preserves_masks_and_changes_attention_sharpness(self):
        logits = torch.tensor([[[[0.0, math.log(3), -float("inf")]]]])
        neutral = AttentionTemperature().run(logits).tensors["attention"]
        sharp = AttentionTemperature(0.5).run(logits).tensors["attention"]
        scaled_logits = AttentionTemperature(0.5).run(logits).tensors["logits"]
        self.assertTrue(torch.isneginf(scaled_logits[0, 0, 0, 2]))
        self.assertAlmostEqual(scaled_logits[0, 0, 0, 1].item(), 2 * math.log(3), places=6)
        torch.testing.assert_close(neutral, torch.tensor([[[[0.25, 0.75, 0.0]]]]))
        torch.testing.assert_close(sharp, torch.tensor([[[[0.1, 0.9, 0.0]]]]))
        for temperature in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                AttentionTemperature(temperature)
        with self.assertRaisesRegex(ValueError, "unmasked key"):
            AttentionTemperature().run(torch.full((1, 1, 1, 2), -float("inf")))
        padded = AttentionTemperature().run(torch.full((1, 1, 1, 2), -float("inf")),
                                             query_mask=torch.tensor([[False]]))
        torch.testing.assert_close(padded.tensors["attention"], torch.zeros(1, 1, 1, 2))


if __name__ == "__main__":
    unittest.main()
