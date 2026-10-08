"""Real native eager kernels: forward/gradient parity and consumed interventions."""
import importlib.util
from importlib.metadata import version
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from hf_model_matrix import make_model
from vlm_probing import Prober, TokenMargin

HAS_V5 = (importlib.util.find_spec("transformers") is not None
          and version("transformers").split(".")[:2] == ["5", "3"])


def make_pixtral():
    from transformers import LlavaConfig, LlavaForConditionalGeneration
    text = dict(model_type="mistral", vocab_size=40, hidden_size=16, intermediate_size=24,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                head_dim=8, pad_token_id=0, sliding_window=4, attention_dropout=0.25)
    vision = dict(model_type="pixtral", hidden_size=16, intermediate_size=24,
                  num_hidden_layers=2, num_attention_heads=2, image_size=4, patch_size=2)
    config = LlavaConfig(text_config=text, vision_config=vision, image_token_index=28,
                         image_seq_length=4, vision_feature_select_strategy="full", vision_feature_layer=-1)
    config._attn_implementation = "eager"
    inputs = {"input_ids": torch.tensor([[1, 28, 28, 28, 28, 3]]),
              "attention_mask": torch.ones(1, 6, dtype=torch.long),
              "pixel_values": torch.randn(1, 3, 4, 4), "image_sizes": torch.tensor([[4, 4]])}
    return LlavaForConditionalGeneration(config).eval(), inputs


@unittest.skipUnless(HAS_V5, "audited native taps require Transformers 5.3.x")
class ConsumedAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(67)
        torch.set_num_threads(1)

    def check_native(self, family):
        model, inputs = make_pixtral() if family == "pixtral" else make_model(family)
        # Native derivatives are measured before installing a probe wrapper.
        model.eval().requires_grad_(True)
        native = model(**inputs, use_cache=False).logits
        loss = native[:, -1, 4].sum()
        embedding = model.get_input_embeddings().weight
        native_gradient, = torch.autograd.grad(loss, [embedding])
        native_logits = native.detach()
        weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
        probe = Prober(model)
        with torch.enable_grad():
            tapped = model(**inputs, use_cache=False).logits
            tapped_gradient, = torch.autograd.grad(tapped[:, -1, 4].sum(), [embedding])
        torch.testing.assert_close(tapped.detach(), native_logits, rtol=0, atol=0)
        torch.testing.assert_close(tapped_gradient, native_gradient, rtol=0, atol=0)
        self.assertEqual(weights.keys(), model.state_dict().keys())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
        model.requires_grad_(False)
        layer = min(probe.spec.editable_attention)
        site = probe.spec.attentions[layer]
        before = [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in model.modules()]
        native_trace = probe._run(inputs, capture=[site])
        actual = probe._run(inputs, capture=[site], interventions={site: lambda value: value*0})
        self.assertEqual(actual.activations[site].abs().sum().item(), 0)
        self.assertGreater((actual.logits-native_trace.logits).abs().max().item(), 1e-7)
        identity = probe._run(inputs, interventions={site: lambda value: value})
        torch.testing.assert_close(identity.logits, native_logits, rtol=0, atol=0)
        probe.attention.attribution(layers=[layer], steps=3, queries="last_prompt", keys="visual").run(
            inputs, metric=TokenMargin(4, 5))
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            probe._run(inputs, interventions={site:
                lambda _: (_ for _ in ()).throw(RuntimeError("deliberate"))})
        decoder_config = model.get_submodule(probe.adapter.sites[site].module.rsplit(".", 1)[0]).config
        self.assertEqual(decoder_config._attn_implementation, "eager")
        self.assertEqual(before, [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in model.modules()])
        torch.testing.assert_close(probe._run(inputs).logits, native_logits, rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_internvl_qwen2(self):
        self.check_native("internvl")

    def test_qwen3_vl_normalized_qk(self):
        self.check_native("qwen3_vl")

    def test_qwen3_5_full_attention_gate(self):
        self.check_native("qwen3_5")

    def test_qwen2_vl_multimodal_rotary(self):
        self.check_native("qwen2_vl")

    def test_qwen2_5_vl_multimodal_rotary(self):
        self.check_native("qwen2_5_vl")

    def test_pixtral_mistral(self):
        self.check_native("pixtral")

    def test_mistral_training_dropout_parity(self):
        # Probes intentionally evaluate, but their persistent Identity wrappers
        # must also preserve a caller's ordinary train-mode dropout behavior.
        model, inputs = make_pixtral()
        model.train()
        torch.manual_seed(91)
        native = model(**inputs, use_cache=False).logits
        probe = Prober(model)
        torch.manual_seed(91)
        actual = model(**inputs, use_cache=False).logits
        torch.testing.assert_close(actual, native, rtol=0, atol=0)
        self.assertTrue(model.training)
        self.assertTrue(probe.spec.editable_attention)


if __name__ == "__main__":
    unittest.main()
