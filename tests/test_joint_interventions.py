"""Native numerical checks for live multi-layer interventions and hook cleanup."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from hf_model_matrix import make_model
from vlm_probing import Prober, TokenMargin, TokenLayout


class JointInterventionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(43)
        self.model, self.inputs = make_model("llava")
        self.model.eval()
        with torch.no_grad():
            self.original = self.model(**self.inputs, use_cache=False).logits
        self.weights = {key: value.clone() for key, value in self.model.state_dict().items()}
        self.probe = Prober(self.model)

    def test_score_tap_preserves_native_forward_and_weights(self):
        torch.testing.assert_close(self.probe._run(self.inputs).logits, self.original, rtol=0, atol=0)
        self.assertEqual(self.weights.keys(), self.model.state_dict().keys())
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, self.weights[key], rtol=0, atol=0)

    def test_joint_knockout_masks_real_probabilities_at_every_layer(self):
        sites = list(self.probe.spec.attentions.values())
        method = self.probe.causal.knockout(layers=[0, 1], queries="last_prompt", keys="visual", joint=True)
        trace = method.forward(self.inputs, capture=sites)
        layout = self.probe._layout(self.inputs, trace.logits)
        blocked = layout.select("last_prompt")[:, None, :, None] & layout.select("visual")[:, None, None, :]
        for site in sites:
            weights = trace.activations[site]
            self.assertEqual(weights[blocked.expand_as(weights)].abs().sum().item(), 0)
            torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
        self.assertGreater((trace.logits - self.original).abs().max().item(), 1e-5)
        self.assertEqual(self.model.model.language_model.config._attn_implementation, "eager")

    def test_joint_steering_edits_live_downstream_stream(self):
        p = self.probe
        direction = torch.arange(16, dtype=torch.float64)[None, None] / 10
        first = p.causal.steer(layers=[0], tokens="all", joint=True).forward(self.inputs, direction=direction)
        second = p.causal.steer(layers=[1], tokens="all", joint=True).forward(self.inputs, direction=direction)
        both = p.causal.steer(layers=[0, 1], tokens="all", joint=True).forward(self.inputs, direction=direction)
        self.assertFalse(torch.allclose(both.logits, first.logits))
        self.assertFalse(torch.allclose(both.logits, second.logits))
        zero = p.causal.steer(layers=[0, 1], tokens="all", strength=0, joint=True).run(
            self.inputs, direction=direction, metric=TokenMargin(4, 5))
        torch.testing.assert_close(zero.tensors["effect"], torch.zeros(2), rtol=0, atol=0)
        self.assertTrue(zero.metadata["joint"])

    def test_exception_removes_hooks_and_restores_backend(self):
        p = self.probe
        attn = self.model.model.language_model.layers[0].self_attn
        counts = [len(module._forward_hooks) + len(module._forward_pre_hooks) for module in self.model.modules()]
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            p._run(self.inputs, interventions={p.spec.attention_scores[0]:
                lambda _: (_ for _ in ()).throw(RuntimeError("deliberate"))})
        self.assertEqual(attn.config._attn_implementation, "eager")
        self.assertEqual(counts, [len(module._forward_hooks) + len(module._forward_pre_hooks)
                                  for module in self.model.modules()])
        torch.testing.assert_close(p._run(self.inputs).logits, self.original, rtol=0, atol=0)

    def test_prediction_scope_and_greedy_replay(self):
        valid = torch.ones(1, 6, dtype=torch.bool)
        prompt = valid.clone()
        prompt[:, 4:] = False
        layout = TokenLayout(valid, prompt=prompt)
        self.assertEqual(layout.select("prediction").nonzero().tolist(), [[0, 3], [0, 4], [0, 5]])
        # Greedy no-cache reference on one real multimodal example.
        values = {key: value[1:2] for key, value in self.inputs.items()}
        values["input_ids"] = values["input_ids"][:, values["attention_mask"][0].bool()]
        values["attention_mask"] = torch.ones_like(values["input_ids"])
        p = self.probe
        method = p.causal.steer(layers=[0, 1], tokens="prediction", strength=0, joint=True)
        actual = method.generate(values, direction=torch.ones(16), max_new_tokens=3)
        original_length = values["input_ids"].shape[1]
        for _ in range(actual.tensors["generated_ids"].shape[1]):
            token = p._run(values).logits[:, -1].argmax(-1, keepdim=True)
            values = {**values, "input_ids": torch.cat((values["input_ids"], token), 1),
                      "attention_mask": torch.cat((values["attention_mask"],
                                                   torch.ones_like(values["attention_mask"][:, :1])), 1)}
        torch.testing.assert_close(actual.tensors["generated_ids"], values["input_ids"][:, original_length:])

    def test_finite_causal_mask_cannot_become_a_legal_future_key(self):
        values = {key: value[:1] for key, value in self.inputs.items()}
        values["input_ids"] = values["input_ids"][:, values["attention_mask"][0].bool()]
        values["attention_mask"] = torch.ones_like(values["input_ids"])
        with self.assertRaisesRegex(ValueError, "no legal key"):
            self.probe.causal.knockout(layers=[0], queries=[0], keys=[0], joint=True).forward(values)


if __name__ == "__main__":
    unittest.main()
