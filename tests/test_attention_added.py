"""Mathematical and source-parity checks for added attention attribution kernels."""

import unittest
import torch

from vlm_probing.attention.attribution import AttentionAttribution
from vlm_probing.attention.beyond_intuition import BeyondIntuition
from vlm_probing.attention.grad_cam import AttentionGradCAM
from vlm_probing.attention.tam import TransitionAttentionMaps


def probabilities(layers=3, batch=2, heads=2, tokens=3):
    generator = torch.Generator().manual_seed(42)
    return torch.rand(layers, batch, heads, tokens, tokens,
                      dtype=torch.float64, generator=generator).softmax(-1)


class AddedAttentionTests(unittest.TestCase):
    def test_gradcam_pools_selected_keys_per_head_before_rectifying(self):
        attention = torch.tensor([[[[.1, .6, .3]], [[.2, .2, .6]]]], dtype=torch.float64)
        gradient = torch.tensor([[[[100., 4., -2.]], [[-100., -2., -4.]]]], dtype=torch.float64)
        keys = torch.tensor([[False, True, True]])
        result = AttentionGradCAM().run(attention, gradient, key_mask=keys)
        torch.testing.assert_close(result.tensors["head_weights"], torch.tensor([[[1.], [-3.]]], dtype=torch.float64))
        # Raw final CLS attention remains in the full row but not in key pooling.
        torch.testing.assert_close(result.tensors["cam"], torch.zeros(1, 1, 3, dtype=torch.float64))
        elementwise = (attention * gradient).clamp_min(0).mean(1)
        assert elementwise[0, 0, 1] > 0  # This is deliberately not gradient x A.


    def test_gradcam_official_cls_baseline_and_constant_normalization(self):
        attention = probabilities(layers=1, batch=1, tokens=4)[0]
        gradient = torch.arange(attention.numel(), dtype=torch.float64).reshape_as(attention) - 7
        qmask = torch.tensor([[True, False, False, False]])
        kmask = torch.tensor([[False, True, True, True]])
        result = AttentionGradCAM().run(attention, gradient, query_mask=qmask, key_mask=kmask)
        official = (attention[0, :, 0, 1:] * gradient[0, :, 0, 1:].mean(-1, keepdim=True)).mean(0).clamp_min(0)
        torch.testing.assert_close(result.tensors["cam"][0, 0, 1:], official)
        assert not result.tensors["cam"][0, 1:].any()
        constant = AttentionGradCAM().run(torch.ones(1, 1, 1, 3, dtype=torch.float64) / 3,
                                        torch.ones(1, 1, 1, 3, dtype=torch.float64), normalize=True)
        assert not constant.tensors["cam"].any()


    def test_hao_ig_actual_attention_path_recovers_nonlinear_difference(self):
        attention = probabilities(layers=1, batch=1)[0]
        coefficients = torch.arange(attention.numel(), dtype=torch.float64).reshape_as(attention) - 7
        # Explicit F(A) = sum(coefficients*A**2). Scale all heads together;
        # uniform midpoint integration is exact for this linear gradient.
        steps = 10
        gradients = []
        for alpha in (torch.arange(steps, dtype=torch.float64) + .5) / steps:
            scaled = (alpha * attention).detach().requires_grad_()
            gradient, = torch.autograd.grad((coefficients * scaled.square()).sum(), scaled)
            gradients.append(gradient)
        integrated = torch.stack(gradients).mean(0)
        result = AttentionAttribution().run(attention, integrated, steps=steps)
        torch.testing.assert_close(result.tensors["attribution"].sum(), (coefficients * attention.square()).sum())
        endpoint = AttentionAttribution().run(attention, 2 * coefficients * attention)
        torch.testing.assert_close(endpoint.tensors["attribution"], result.tensors["attribution"] * 2)
        assert result.tensors["attribution"].min() < 0


    def test_hao_left_and_right_quadrature_differ_as_paper_and_author_code(self):
        attention = probabilities(layers=1, batch=1)[0]
        steps = 4
        right = (2 * attention.unsqueeze(0) * (torch.arange(1, steps + 1, dtype=attention.dtype) / steps)[:, None, None, None, None]).mean(0)
        left = (2 * attention.unsqueeze(0) * (torch.arange(steps, dtype=attention.dtype) / steps)[:, None, None, None, None]).mean(0)
        r = AttentionAttribution().run(attention, right, steps=steps, quadrature="right")
        l = AttentionAttribution().run(attention, left, steps=steps, quadrature="left")
        torch.testing.assert_close(r.tensors["attribution"] - l.tensors["attribution"], 2 / steps * attention.square())


    def test_tam_backward_transition_matches_official_code_up_to_global_factor(self):
        attention = probabilities(batch=1)
        gradient = torch.linspace(-2, 3, attention[0].numel(), dtype=torch.float64).reshape_as(attention[0])
        mean_attention = attention.mean(2)
        official = mean_attention[-1][:, :1]
        for layer in reversed(mean_attention[:-1]):
            official = official + official @ layer
        feedback = gradient.clamp_min(0).mean(1)[:, :1]
        result = TransitionAttentionMaps().run(attention, gradient)
        unnormalized = TransitionAttentionMaps(residual_normalize=False).run(attention, gradient)
        torch.testing.assert_close(unnormalized.tensors["relevance"][:, :1], official * feedback)
        torch.testing.assert_close(result.tensors["relevance"], unnormalized.tensors["relevance"] / 4)
        torch.testing.assert_close(result.tensors["state"].sum(-1), torch.ones(1, 3, dtype=torch.float64))
        # Correct orientation is last-layer state @ earlier-layer transitions.
        wrong = mean_attention[0] @ mean_attention[-1]
        assert not torch.allclose(result.tensors["state"], wrong)


    def test_tam_rectifies_after_input_integration_before_averaging_heads(self):
        attention = probabilities(layers=1, batch=1)
        integrated = torch.full_like(attention[0], -2)
        integrated[:, 0] = 1
        result = TransitionAttentionMaps().run(attention, integrated)
        torch.testing.assert_close(result.tensors["feedback"], torch.full((1, 3, 3), .5, dtype=torch.float64))
        assert not integrated.mean(1).clamp_min(0).any()
        # With one layer there is no earlier residual transition.
        torch.testing.assert_close(result.tensors["state"], attention[0].mean(1))


    def test_beyond_intuition_head_matches_official_single_example(self):
        attention = probabilities(batch=1)
        gradient = torch.arange(attention.numel(), dtype=torch.float64).reshape_as(attention) - 15
        final_integrated = gradient[-1] / 3
        official = torch.eye(3, dtype=torch.float64)
        expected_weights = []
        for a, g in zip(attention[:, 0], gradient[:, 0]):
            importance = (a.transpose(-2, -1) @ g).abs().mean((-2, -1))
            weights = importance / importance.sum()
            expected_weights.append(weights)
            c = (weights[:, None, None] * a).sum(0)
            official = official + c @ official
        result = BeyondIntuition().run(attention, final_integrated, gradients=gradient)
        torch.testing.assert_close(result.tensors["perception"][0], official)
        torch.testing.assert_close(result.tensors["head_weights"][:, 0], torch.stack(expected_weights))
        torch.testing.assert_close(result.tensors["relevance"][0], official * final_integrated[0].clamp_min(0).mean(0))


    def test_beyond_intuition_head_keeps_examples_independent_and_handles_zero_gradient(self):
        attention = probabilities()
        gradients = torch.randn_like(attention)
        gradients[:, 1] *= 1e6
        result = BeyondIntuition().run(attention, gradients[-1], gradients=gradients)
        individual = BeyondIntuition().run(attention[:, :1], gradients[-1, :1], gradients=gradients[:, :1])
        torch.testing.assert_close(result.tensors["relevance"][:1], individual.tensors["relevance"])
        zero = BeyondIntuition().run(attention, torch.ones_like(attention[0]), gradients=torch.zeros_like(attention))
        torch.testing.assert_close(zero.tensors["perception"], torch.eye(3, dtype=torch.float64).expand(2, 3, 3))
        assert not zero.tensors["head_weights"].any()


    def test_beyond_intuition_token_scales_source_columns_using_value_projection_norms(self):
        attention = probabilities(layers=1, batch=1)
        inputs = torch.tensor([[[2., 4., 8.]]], dtype=torch.float64)
        projected = torch.tensor([[[4., 4., 2.]]], dtype=torch.float64)
        weights = projected[0] / inputs[0]
        result = BeyondIntuition(variant="token").run(attention, torch.ones_like(attention[0]),
                                                      input_norms=inputs, projected_value_norms=projected)
        expected = torch.eye(3, dtype=torch.float64)[None] + attention[0].mean(1) @ torch.diag_embed(weights)
        torch.testing.assert_close(result.tensors["perception"], expected)
        torch.testing.assert_close(result.tensors["token_weights"], weights[None])
        row_scaled = torch.eye(3, dtype=torch.float64)[None] + weights[:, :, None] * attention[0].mean(1)
        assert not torch.allclose(expected, row_scaled)


    def check_padding(self, method):
        attention = probabilities(layers=2, batch=1, tokens=2)
        gradients = torch.randn_like(attention)
        padded = torch.zeros(2, 1, 2, 3, 3, dtype=torch.float64)
        padded[..., :2, :2] = attention
        pg = torch.zeros_like(padded)
        pg[..., :2, :2] = gradients
        mask = torch.tensor([[True, True, False]])
        if method == "tam":
            kernel = TransitionAttentionMaps()
            a = kernel.run(attention, gradients[-1])
            p = kernel.run(padded, pg[-1], valid_tokens=mask)
        elif method == "head":
            kernel = BeyondIntuition()
            a = kernel.run(attention, gradients[-1], gradients=gradients)
            p = kernel.run(padded, pg[-1], gradients=pg, valid_tokens=mask)
        else:
            kernel = BeyondIntuition(variant="token")
            a = kernel.run(attention, gradients[-1], input_norms=torch.ones(2, 1, 2, dtype=torch.float64),
                           projected_value_norms=torch.ones(2, 1, 2, dtype=torch.float64))
            # Padding norms can be zero: only valid residual tokens need positivity.
            norms = mask.double()[None].expand(2, 1, 3)
            p = kernel.run(padded, pg[-1], input_norms=norms, projected_value_norms=norms, valid_tokens=mask)
        torch.testing.assert_close(a.tensors["relevance"], p.tensors["relevance"][..., :2, :2])
        assert not p.tensors["relevance"][..., 2, :].any()
        assert not p.tensors["relevance"][..., :, 2].any()


    def test_malformed_inputs_fail_before_silent_broadcast_or_wrong_path_claims(self):
        attention = probabilities(layers=1)
        with self.assertRaisesRegex(ValueError, "match attention"):
            AttentionGradCAM().run(attention[0], torch.ones(1, 2, 3, 3, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "selected key"):
            AttentionGradCAM().run(attention[0], torch.ones_like(attention[0]), key_mask=torch.zeros(2, 3, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "start_layer"):
            TransitionAttentionMaps(start_layer=1).run(attention, torch.ones_like(attention[0]))
        with self.assertRaisesRegex(ValueError, "positive"):
            BeyondIntuition(variant="token").run(attention, torch.ones_like(attention[0]),
                                                   input_norms=torch.zeros(1, 2, 3, dtype=torch.float64),
                                                   projected_value_norms=torch.ones(1, 2, 3, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "quadrature"):
            AttentionAttribution().run(attention[0], torch.ones_like(attention[0]), quadrature="gradient_x_attention")

    def test_padding_all_variants(self):
        for method in ("tam", "head", "token"):
            with self.subTest(method=method):
                self.check_padding(method)


if __name__ == "__main__":
    unittest.main()
