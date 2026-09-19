"""Optional architecture integration; tiny random models, no network or weights."""
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from hf_smoke import make_llama, make_qwen
from vlm_probing import CapabilityError, Prober, TokenMargin


@unittest.skipUnless(importlib.util.find_spec("transformers"), "optional transformers extra not installed")
class HuggingFaceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)

    def test_llama_padding_gqa_and_frozen_relevance(self):
        model, inputs = make_llama()
        model.requires_grad_(False)
        probe = Prober(model)
        result = probe.lens.logit(layers=[1]).run(inputs)
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits
        torch.testing.assert_close(result.tensors["logits"][0], logits[[0, 1], [2, 3]])
        self.assertEqual(result.tensors["positions"].tolist(), [[0, 2], [1, 3]])
        probe.attention.relevance().run(inputs, metric=TokenMargin(4, 5))
        site = probe.spec.heads[0]
        raw = probe.adapter.run({**inputs, "use_cache": False}, capture=[site]).activations[site]
        projected = probe._heads(0, raw)
        torch.testing.assert_close(projected.sum(-2), model.model.layers[0].self_attn.o_proj(raw))
        self.assertFalse(probe.describe()["methods"]["attention.reweight"]["available"])
        with self.assertRaises(CapabilityError):
            probe.attention.reweight(layers=[0])
        with self.assertRaises(CapabilityError):
            probe.causal.knockout(layers=[0])

    def test_qwen_actual_image_path_unequal_grids_and_six_lenses(self):
        model, inputs = make_qwen()
        probe = Prober(model)
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits
        result = probe.lens.logit(layers=[1]).run(inputs)
        torch.testing.assert_close(result.tensors["logits"][0], logits[[0, 1], [4, 5]])
        visual = probe.lens.embed().run(inputs, top_k=2)
        self.assertEqual(visual.tensors["positions"].tolist(), [[0, 2], [1, 2], [1, 3]])
        self.assertEqual(visual.tensors["token_ids"].shape, (1, 3, 2))
        probe.lens.tuned(layers=[0]).fit(inputs, steps=2).run(inputs)
        probe.lens.attention(layers=[0]).fit(inputs, steps=2).run(inputs)
        probe.lens.jacobian(layers=[0]).fit(inputs, skip_first=0, exclude_last=False).run(inputs)
        scoped = probe.lens.patchscope(layers=[0], source_position=2, target_layer=0, target_position=2).run(
            inputs, target_inputs=inputs)
        torch.testing.assert_close(scoped.tensors["logits"][0], logits)
        metric = TokenMargin(4, 5)
        bad = {**inputs, "pixel_values": torch.zeros_like(inputs["pixel_values"])}
        patched = probe.causal.patch(layers=[0]).run(bad, source=inputs, metric=metric)
        self.assertTrue((patched.tensors["effect"].abs() > 1e-6).any())
        noop = probe.causal.patch(layers=[0]).run(inputs, source=inputs, metric=metric)
        torch.testing.assert_close(noop.tensors["effect"], torch.zeros(1, 2))
        probe.attention.profile().run(inputs)
        probe.attention.rollout().run(inputs)
        probe.attention.relevance().run(inputs, metric=metric)
        probe.attention.head_logits().run(inputs)

    def test_alignment_grid_check_and_explicit_cache_rejection(self):
        model, inputs = make_qwen()
        probe = Prober(model)
        # Same visual token count but different spatial coordinates.
        changed = {**inputs, "image_grid_thw": torch.tensor([[1, 2, 2], [1, 4, 2]])}
        with self.assertRaisesRegex(ValueError, "image_grid_thw"):
            probe.causal.patch(layers=[0]).run(inputs, source=changed, metric=TokenMargin(4, 5))
        with self.assertRaisesRegex(ValueError, "cache"):
            probe.lens.logit().run({**inputs, "use_cache": True})

    def test_sdpa_does_not_claim_attention_observation(self):
        model, inputs = make_llama()
        model.set_attn_implementation("sdpa")
        probe = Prober(model)
        self.assertFalse(probe.describe()["methods"]["attention.profile"]["available"])
        with self.assertRaises(CapabilityError):
            probe.attention.profile()
        probe.lens.logit().run(inputs)
        probe.lens.attention(layers=[0]).fit(inputs, steps=1).run(inputs)


if __name__ == "__main__":
    unittest.main()
