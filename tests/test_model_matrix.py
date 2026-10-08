"""Numerical integration contracts over native, tiny multimodal architectures."""
import importlib.util
from importlib.metadata import version
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from hf_model_matrix import make_model
from vlm_probing import CapabilityError, Prober, TokenMargin

HAS_V5 = (importlib.util.find_spec("transformers") is not None
          and version("transformers").split(".")[:2] == ["5", "3"])


@unittest.skipUnless(HAS_V5, "expanded matrix requires Transformers 5.3.x")
class ModelMatrixTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def check_family(self, family):
        model, inputs = make_model(family)
        model.requires_grad_(False)
        p = Prober(model)
        with torch.no_grad():
            expected = model(**inputs, use_cache=False).logits
        # Transformers installs persistent output-recording hooks lazily on the
        # first CLIP forward. Preserve those upstream hooks as well as user hooks.
        hooks_before = [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in model.modules()]
        last = max(p.spec.residuals)
        result = p.lens.logit(layers=[last]).run(inputs)
        positions = result.tensors["positions"]
        torch.testing.assert_close(result.tensors["logits"][0], expected[positions[:, 0], positions[:, 1]])
        visual = p.spec.layout(inputs).visual
        decoded = p.lens.embed().run(inputs, top_k=2)
        self.assertEqual(decoded.tensors["positions"].tolist(), visual.nonzero().tolist())
        attention_layer = min(p.spec.heads)
        head_site, final_site = p.spec.heads[attention_layer], p.spec.residuals[last]
        trace = p._run(inputs, capture=[head_site, final_site])
        raw, full = trace.activations[head_site], trace.activations[final_site]
        module = model.get_submodule(p.adapter.sites[head_site].module)
        projected = p._heads(attention_layer, raw)
        expected_heads = module(raw)
        if module.bias is not None:
            expected_heads = expected_heads - module.bias
        torch.testing.assert_close(projected.sum(-2), expected_heads)
        weight, scale, center = p.spec.linear_readout(full)
        self.assertFalse(center)
        torch.testing.assert_close(torch.nn.functional.linear(full * scale, weight), expected,
                                   atol=2e-6, rtol=1e-4)
        p.lens.tuned(layers=[0]).fit(inputs, steps=2).run(inputs)
        p.lens.attention(layers=[attention_layer]).fit(inputs, steps=2).run(inputs)
        p.lens.jacobian(layers=[0]).fit(inputs, skip_first=0, exclude_last=False).run(inputs)
        p.lens.patchscope(layers=[0], source_position=2, target_layer=0, target_position=2).run(
            inputs, target_inputs=inputs)
        metric = TokenMargin(4, 5)
        noop = p.causal.patch(layers=[0]).run(inputs, source=inputs, metric=metric)
        torch.testing.assert_close(noop.tensors["effect"], torch.zeros(1, 2))
        # SmolVLM treats all-zero tiles as padding. Invert image values instead.
        corrupt = {**inputs, "pixel_values": -inputs["pixel_values"]}
        patched = p.causal.patch(layers=[0]).run(corrupt, source=inputs, metric=metric)
        self.assertTrue((patched.tensors["effect"].abs() > 1e-7).any(), family)
        if family == "qwen3_5":
            self.assertFalse(p.describe()["methods"]["causal.path"]["available"])
            with self.assertRaisesRegex(CapabilityError, "EVERY"):
                p.causal.path(senders=[(3, 0)])
        else:
            for receivers in ("residual", [(1, 0, "q")], [(1, 0, "k")], [(1, 0, "v")]):
                tokens = "visual" if isinstance(receivers, list) and receivers[0][2] in {"k", "v"} else "last_prompt"
                path = p.causal.path(senders=[(0, 0)], receivers=receivers,
                                     sender_tokens=tokens, receiver_tokens=tokens)
                identity = path.run(inputs, donor=inputs, metric=metric)
                torch.testing.assert_close(identity.tensors["effect"], torch.zeros(2))
                effect = path.run(inputs, donor=corrupt, metric=metric)
                self.assertTrue((effect.tensors["effect"].abs() > 1e-9).any(), (family, receivers))
        p.causal.attribute(layers=[0]).run(corrupt, source=inputs, metric=metric)
        p.causal.steer(layers=[0]).run(inputs, direction=torch.ones(16) * 0.1, metric=metric)
        if p.spec.attention_scores:
            knockout = p.causal.knockout(layers=sorted(p.spec.attention_scores)[:2], queries="last_prompt",
                                         keys="visual", joint=True).run(inputs, metric=metric)
            self.assertTrue((knockout.tensors["effect"].abs() > 1e-7).any())
        p.attention.profile().run(inputs)
        p.attention.head_logits().run(inputs)
        if family == "qwen3_5":
            self.assertEqual(p.describe()["methods"]["attention.profile"]["sites"], [3])
            self.assertFalse(p.describe()["methods"]["attention.rollout"]["available"])
            for factory in (p.attention.rollout, p.attention.relevance):
                with self.assertRaises(CapabilityError):
                    factory()
                with self.assertRaises(CapabilityError):
                    factory(layers=[3])
        else:
            p.attention.rollout().run(inputs)
            p.attention.relevance().run(inputs, metric=metric)
        self.assertTrue(all(param.grad is None for param in model.parameters()))
        self.assertEqual(hooks_before, [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks))
                                        for m in model.modules()])

    def test_qwen3_5(self):
        self.check_family("qwen3_5")

    def test_qwen3_vl(self):
        self.check_family("qwen3_vl")

    def test_qwen2_5_vl(self):
        self.check_family("qwen2_5_vl")

    def test_qwen2_vl(self):
        self.check_family("qwen2_vl")

    def test_smolvlm(self):
        self.check_family("smolvlm")

    def test_internvl(self):
        self.check_family("internvl")

    def test_llava(self):
        self.check_family("llava")

    def test_deepstack_residual_includes_visual_injection(self):
        model, inputs = make_model("qwen3_vl")
        p = Prober(model)
        block_outputs = []
        handle = model.model.language_model.layers[0].register_forward_hook(
            lambda _module, _args, output: block_outputs.append(output.detach().clone()))
        try:
            trace = p._run(inputs, capture=[p.spec.residuals[0]])
        finally:
            handle.remove()
        difference = trace.activations[p.spec.residuals[0]] - block_outputs[0]
        visual = p.spec.layout(inputs).visual
        self.assertTrue(difference[visual].abs().max() > 0)
        torch.testing.assert_close(difference[~visual], torch.zeros_like(difference[~visual]))


if __name__ == "__main__":
    unittest.main()
