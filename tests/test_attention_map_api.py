"""Actual target-conditioned forwards and interventions, without downloads."""
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

from vlm_probing import CapabilityError, ClassScore, Prober

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))


@unittest.skipUnless(importlib.util.find_spec("transformers"), "optional transformers extra")
class AttentionMapAPITests(unittest.TestCase):
    def setUp(self):
        from transformers import ViTConfig, ViTForImageClassification
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.model = ViTForImageClassification(ViTConfig(
            image_size=16, patch_size=8, hidden_size=16, num_hidden_layers=2,
            num_attention_heads=2, intermediate_size=32, num_labels=3,
            attn_implementation="eager")).eval()
        self.inputs = {"pixel_values": torch.randn(1, 3, 16, 16)}
        with torch.no_grad():
            self.original = self.model(**self.inputs).logits.clone()
        self.keys = set(self.model.state_dict())
        self.probe = Prober(self.model)
        self.metric = ClassScore(1)

    def test_taps_preserve_native_outputs_and_weights_and_edit_consumed_probabilities(self):
        self.assertEqual(self.keys, set(self.model.state_dict()))
        torch.testing.assert_close(self.model(**self.inputs).logits, self.original, rtol=0, atol=0)
        noop = self.probe.attention.reweight(layers=[0], queries=[0], keys="visual", weight=1).run(
            self.inputs, metric=self.metric)
        torch.testing.assert_close(noop.tensors["effect"], torch.zeros(1, 1), rtol=0, atol=1e-7)
        effect = self.probe.attention.reweight(layers=[0], queries=[0], keys="visual", weight=0).run(
            self.inputs, metric=self.metric).tensors["effect"]
        self.assertGreater(effect.abs().max().item(), 1e-6)

    def test_attention_ig_integrates_fixed_layer_and_preserves_signed_completeness(self):
        self.model.requires_grad_(False)
        method = self.probe.attention.attribution(layers=[0], queries="all", keys="all", steps=64)
        result = method.run(self.inputs, metric=self.metric)
        site = self.probe.spec.attentions[0]
        zero = self.probe._run(self.inputs, interventions={site: torch.zeros_like})
        difference = self.original[:, 1] - zero.logits[:, 1]
        total = result.tensors["attribution"][0].sum((1, 2, 3))
        torch.testing.assert_close(total, difference, rtol=.03, atol=3e-5)
        self.assertTrue((result.tensors["attribution"] < 0).any())
        self.assertFalse(result.metadata["layer_results"][0]["row_renormalized"])
        self.assertTrue(all(x.grad is None for x in self.model.parameters()))
        with torch.no_grad():
            repeated = method.run(self.inputs, metric=self.metric)
        torch.testing.assert_close(repeated.tensors["attribution"], result.tensors["attribution"])

    def test_input_integrals_match_explicit_reruns_and_target_changes_maps(self):
        self.model.requires_grad_(False)
        sites = [self.probe.spec.attentions[i] for i in (0, 1)]
        base = self.probe._run(self.inputs, capture=sites)
        integrated = []
        for alpha in (.5, 1.):
            inp = {"pixel_values": (alpha*self.inputs["pixel_values"]).requires_grad_(True)}
            trace = self.probe._run(inp, capture=[sites[-1]], grad=True)
            gradient, = torch.autograd.grad(trace.logits[:, 1].sum(), trace.activations[sites[-1]])
            integrated.append(gradient.detach())
        from vlm_probing.attention import TransitionAttentionMaps
        expected = TransitionAttentionMaps().run(torch.stack([base.activations[s] for s in sites]),
                                                 torch.stack(integrated).mean(0))
        result = self.probe.attention.tam(queries=[0], keys="visual", steps=2).run(
            self.inputs, metric=self.metric)
        torch.testing.assert_close(result.tensors["relevance"], expected.tensors["relevance"])
        self.assertEqual(result.metadata["quadrature"], "right")
        self.assertEqual(result.metadata["integration_steps"], 2)
        a = self.probe.attention.grad_cam(layers=[1], queries=[0], keys="visual").run(
            self.inputs, metric=ClassScore(0)).tensors["head_weights"]
        b = self.probe.attention.grad_cam(layers=[1], queries=[0], keys="visual").run(
            self.inputs, metric=ClassScore(1)).tensors["head_weights"]
        self.assertGreater((a-b).abs().max().item(), 1e-8)
        default = self.probe.attention.grad_cam(layers=[1]).run(self.inputs, metric=self.metric)
        self.assertEqual(default.tensors["query_mask"].nonzero().tolist(), [[0, 0]])
        self.assertEqual(default.tensors["key_mask"].nonzero().tolist(), [[0, 1], [0, 2], [0, 3], [0, 4]])

    def test_beyond_intuition_token_variant_uses_v_projection_not_head_output(self):
        method = self.probe.attention.beyond_intuition(variant="token", queries=[0], keys="visual", steps=2)
        result = method.run(self.inputs, metric=self.metric)
        layer = self.model.vit.encoder.layer[0]
        captures = [self.probe.spec.attention_inputs[0], self.probe.spec.attention_values[0]]
        trace = self.probe._run(self.inputs, capture=captures)
        raw, value = [trace.activations[s] for s in captures]
        expected = torch.nn.functional.linear(value, layer.attention.output.dense.weight).norm(dim=-1) / raw.norm(dim=-1)
        torch.testing.assert_close(result.tensors["token_weights"][0], expected)
        self.assertTrue((result.tensors["map"][:, 1:] == 0).all())
        self.assertTrue((result.tensors["map"][:, :, 0] == 0).all())

    def test_cleanup_after_integration_error_and_classification_capabilities(self):
        self.model.train()
        modes = [m.training for m in self.model.modules()]
        count = 0

        def fail_on_second(logits):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("deliberate metric failure")
            return logits[:, 1]

        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            self.probe.attention.attribution(layers=[0], steps=2).run(self.inputs, metric=fail_on_second)
        self.assertEqual(modes, [m.training for m in self.model.modules()])
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))
        self.assertEqual(self.model.config._attn_implementation, "eager")
        self.assertFalse(self.probe.describe()["methods"]["lens.logit"]["available"])
        with self.assertRaises(CapabilityError):
            self.probe.lens.logit()
        with self.assertRaisesRegex(ValueError, "baseline is only"):
            self.probe.attention.grad_cam().run(self.inputs, metric=self.metric,
                                              baseline=torch.zeros_like(self.inputs["pixel_values"]))

    def test_native_vlm_transfers_respect_geometry_backends_and_hybrid_gaps(self):
        from hf_model_matrix import make_model
        from vlm_probing import TokenMargin
        for family in ("qwen3_5", "qwen3_vl", "qwen2_5_vl", "qwen2_vl", "smolvlm", "internvl", "llava"):
            with self.subTest(family=family):
                model, inputs = make_model(family)
                model.requires_grad_(False)
                probe = Prober(model)
                layer = min(probe.spec.attentions)
                metric = TokenMargin(4, 5)
                result = probe.attention.grad_cam(layers=[layer], keys="visual").run(inputs, metric=metric)
                self.assertTrue(torch.isfinite(result.tensors["cam"]).all())
                raw = probe.attention.profile(layers=[layer], include_attention=True).run(inputs)
                self.assertEqual(raw.tensors["attention"].ndim, 5)
                if family == "qwen3_5":
                    for method in (probe.attention.tam, probe.attention.beyond_intuition):
                        with self.assertRaises(CapabilityError):
                            method()
                    continue
                for method in (probe.attention.tam(steps=2),
                               probe.attention.beyond_intuition(variant="head", steps=2),
                               probe.attention.beyond_intuition(variant="token", steps=2)):
                    result = method.run(inputs, metric=metric)
                    self.assertTrue(torch.isfinite(result.tensors["map"]).all())
                if probe.spec.editable_attention:
                    result = probe.attention.attribution(layers=[layer], steps=2).run(inputs, metric=metric)
                    self.assertTrue(torch.isfinite(result.tensors["attribution"]).all())
                else:
                    with self.assertRaises(CapabilityError):
                        probe.attention.attribution()
                self.assertTrue(all(x.grad is None for x in model.parameters()))


if __name__ == "__main__":
    unittest.main()
