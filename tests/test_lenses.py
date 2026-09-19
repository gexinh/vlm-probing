"""Deterministic CPU tests of the lens algorithms, not paper-level replication."""
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from vlm_probing.core import BaseMethod
from vlm_probing.lenses import (
    AttentionLens, BaseLens, EmbedLens, JacobianLens, LogitLens, Patchscope, TunedLens,
)


BINDING = {"model_id": "toy@1", "site": "layers.0.post", "readout_id": "norm-head@1",
           "tokenizer_id": "toy-vocab@1", "calibration_id": "toy-calibration@1"}


class LensTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(17)

    def test_six_classes_share_a_category_interface(self):
        for method in (LogitLens, EmbedLens, TunedLens, AttentionLens, JacobianLens, Patchscope):
            self.assertTrue(issubclass(method, BaseLens))
            self.assertTrue(issubclass(method, BaseMethod))
            self.assertEqual(method.family, "lens")
        with self.assertRaises(TypeError):
            BaseLens()

    def test_logit_lens_applies_actual_norm_and_head(self):
        norm, head = nn.LayerNorm(3), nn.Linear(3, 5, bias=False)
        readout = nn.Sequential(norm, head)
        states = torch.tensor([[[1., 2., 8.], [4., -2., 3.]]])
        result = LogitLens(readout).run(states, layer=2)
        torch.testing.assert_close(result.tensors["logits"], head(norm(states)))
        self.assertFalse(torch.allclose(result.tensors["logits"], head(states)))
        self.assertEqual(result.metadata["layer"], 2)

    def test_embed_lens_cosine_chunking_and_explicit_groups(self):
        embeddings = torch.tensor([[1., 0.], [0., 7.], [-2., 0.], [0., -1.]])
        states = torch.tensor([[[0.2, 8.], [-3., 0.]]])
        lens = EmbedLens(embeddings, token_groups={"custom_sink": [1]})
        result = lens.run(states, top_k=2, vocabulary_chunk_size=2)
        self.assertEqual(result.tensors["token_ids"][..., 0].tolist(), [[1, 2]])
        self.assertEqual(result.tensors["group_labels"].tolist(), [[0, -1]])
        expected = torch.nn.functional.normalize(states, dim=-1) @ torch.nn.functional.normalize(embeddings, dim=-1).T
        torch.testing.assert_close(result.tensors["similarities"], expected.topk(2, dim=-1).values)

    def test_tuned_lens_improves_kl_without_mutating_readout_or_teacher(self):
        readout = nn.Linear(3, 4)
        readout.train()
        for parameter in readout.parameters():
            parameter.grad = torch.full_like(parameter, 0.75)
        before = {k: v.detach().clone() for k, v in readout.state_dict().items()}
        gradient_before = [p.grad.clone() for p in readout.parameters()]
        states = torch.randn(4, 6, 3, requires_grad=True)
        target = readout(2 * states + torch.tensor([0.5, -1., 2.]))
        target.retain_grad()
        lens = TunedLens(readout, 3, binding=BINDING)
        torch.testing.assert_close(lens.run(states).tensors["logits"], readout(states))
        losses = lens.fit(states, target, steps=120, lr=0.05)
        self.assertLess(losses[-1], losses[0] * 0.02)
        self.assertIsNone(states.grad)
        self.assertIsNone(target.grad)
        self.assertTrue(readout.training)
        for (key, value), gradient, parameter in zip(readout.state_dict().items(), gradient_before, readout.parameters()):
            torch.testing.assert_close(value, before[key])
            torch.testing.assert_close(parameter.grad, gradient)
            self.assertTrue(parameter.requires_grad)

    def test_tuned_lens_mask_ignores_conflicting_unselected_teacher(self):
        states = torch.tensor([[[1., 0.], [1., 0.]]])
        targets = torch.tensor([[[8., -8.], [-8., 8.]]])
        lens = TunedLens(lambda x: x, 2)
        losses = lens.fit(states, targets, steps=80, lr=0.1, mask=torch.tensor([[True, False]]))
        self.assertLess(losses[-1], losses[0] * 0.05)
        self.assertEqual(int(lens.run(states).tensors["logits"][0, 1].argmax()), 0)

    def test_fitting_enables_gradients_and_supports_bf16_readout(self):
        head = nn.Linear(2, 3, dtype=torch.bfloat16)
        states = torch.randn(2, 3, 2)
        with torch.no_grad():
            teacher = head((states * 2).bfloat16())
            lens = TunedLens(head, 2, readout_dtype=torch.bfloat16)
            losses = lens.fit(states, teacher, steps=30, lr=0.1)
            self.assertLess(losses[-1], losses[0])
            self.assertEqual(lens.run(states).tensors["logits"].dtype, torch.bfloat16)
            AttentionLens(1, 2, 3).fit(states.unsqueeze(-2), teacher, steps=2)
        self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_fitting_rejects_inference_mode(self):
        with torch.inference_mode(), self.assertRaises(ValueError):
            TunedLens(lambda x: x, 2).fit(torch.ones(1, 2), torch.ones(1, 2))
        with torch.inference_mode(), self.assertRaises(ValueError):
            JacobianLens(lambda x: x).fit(torch.ones(1, 3, 2), lambda x: x,
                                         skip_first=0, exclude_last=False)

    def test_attention_lens_learns_per_head_maps_from_teacher_distribution(self):
        states = torch.randn(5, 4, 2, 3)
        teacher_maps = torch.randn(2, 3, 4)
        teacher = torch.einsum("bshd,hdv->bsv", states, teacher_maps)
        lens = AttentionLens(2, 3, 4, binding=BINDING)
        losses = lens.fit(states, teacher, steps=140, lr=0.07,
                          mask=torch.ones(5, 4, dtype=torch.bool))
        self.assertLess(losses[-1], losses[0] * 0.02)
        result = lens.run(states)
        self.assertEqual(tuple(result.tensors["head_logits"].shape), (5, 4, 2, 4))
        torch.testing.assert_close(result.tensors["head_logits"].sum(-2), result.tensors["logits"])
        self.assertFalse(torch.allclose(lens.decoders.weight[0], lens.decoders.weight[1]))

    def test_attention_lens_initializes_from_unembedding(self):
        weights, bias = torch.randn(4, 3), torch.randn(4)
        states = torch.randn(1, 2, 2, 3)
        lens = AttentionLens(2, 3, 4, initial_unembedding=weights, initial_bias=bias)
        expected = torch.einsum("bshd,vd->bshv", states, weights) + bias
        torch.testing.assert_close(lens.run(states).tensors["head_logits"], expected)

    def test_attention_default_fit_uses_last_valid_positions(self):
        states = torch.ones(1, 3, 1, 1)
        teacher = torch.tensor([[[-8., 8.], [8., -8.], [-8., 8.]]])
        lens = AttentionLens(1, 1, 2)
        losses = lens.fit(states, teacher, steps=70, lr=0.1,
                          valid_mask=torch.tensor([[True, True, False]]))
        self.assertLess(losses[-1], losses[0] * 0.03)
        self.assertEqual(int(lens.run(states).tensors["logits"][0, 1].argmax()), 0)

    def test_jacobian_causal_future_sum_and_prompt_equal_averaging(self):
        states = torch.randn(2, 4, 2)
        transform = torch.tensor([[2., 1.], [0., 3.]])
        valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
        downstream = lambda x: x.cumsum(1) @ transform.T
        lens = JacobianLens(lambda x: x, binding=BINDING)
        lens.fit(states, downstream, valid_mask=valid, skip_first=0, exclude_last=False)
        # Prompt 1 mean target counts = 2.5, prompt 2 = 1.5, equal-prompt mean = 2.
        torch.testing.assert_close(lens.jacobian, 2 * transform)
        result = lens.run(states, layer=0)
        torch.testing.assert_close(result.tensors["logits"], states @ (2 * transform).T)

    def test_jacobian_sampling_never_uses_invalid_positions(self):
        states = torch.randn(1, 8, 2)
        valid = torch.tensor([[True, True, True, True, True, True, False, False]])
        sources = torch.tensor([[True, False, True, True, True, True, True, True]])
        lens = JacobianLens(lambda x: x)
        lens.fit(states, lambda x: x.cumsum(1), valid_mask=valid, source_mask=sources,
                 skip_first=1, exclude_last=True, max_sources=2, seed=3)
        selected = lens.calibration["source_positions"][0]
        self.assertEqual(len(selected), 2)
        self.assertTrue(set(selected) <= {2, 3, 4})
        self.assertEqual(lens.calibration["target_positions"], [[1, 2, 3, 4]])

    def test_jacobian_does_not_populate_model_gradients(self):
        downstream = nn.Linear(2, 2)
        states = torch.randn(2, 3, 2, requires_grad=True)
        lens = JacobianLens(lambda x: x)
        lens.fit(states, downstream, skip_first=0, exclude_last=False)
        torch.testing.assert_close(lens.jacobian, downstream.weight)
        self.assertTrue(all(p.grad is None for p in downstream.parameters()))
        self.assertIsNone(states.grad)

    def test_patchscope_injects_source_into_prompt_and_uses_target_computation(self):
        calls = []

        def runner(patch, *, target_layer, target_position, target_inputs):
            target = target_inputs["hidden"].clone()
            target[:, target_position] = patch
            calls.append((target_layer, target_position))
            return target.cumsum(1)

        source = torch.tensor([[[2., 3.], [7., 11.]]])
        target = torch.zeros(1, 4, 2)
        lens = Patchscope(runner, source_model_id="toy", target_model_id="toy")
        result = lens.run(source, source_position=1, target_position=2, target_layer=0,
                          target_inputs={"hidden": target}, layer=1)
        self.assertEqual(calls, [(0, 2)])
        torch.testing.assert_close(result.tensors["logits"][0, 3], source[0, 1])
        self.assertEqual(float(target.sum()), 0)
        self.assertEqual(result.metadata["readout"], "teacher_forced_target_prompt")
        with self.assertRaises(ValueError):
            Patchscope(runner, source_model_id="a", target_model_id="b")

    def test_calibrated_artifacts_round_trip_and_reject_other_model(self):
        states = torch.randn(1, 3, 2)
        lenses = [
            (TunedLens(lambda x: x, 2, binding=BINDING), states,
             lambda binding: TunedLens(lambda x: x, 2, binding=binding)),
            (AttentionLens(1, 2, 2, binding=BINDING), states.unsqueeze(-2),
             lambda binding: AttentionLens(1, 2, 2, binding=binding)),
            (JacobianLens(lambda x: x, binding=BINDING).fit(states, lambda x: x * 2,
                                                         skip_first=0, exclude_last=False),
             states, lambda binding: JacobianLens(lambda x: x, binding=binding)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for lens, inputs, factory in lenses:
                path = Path(directory) / (lens.name + ".pt")
                lens.save(path)
                restored = factory(BINDING).load(path)
                torch.testing.assert_close(lens.run(inputs).tensors["logits"], restored.run(inputs).tensors["logits"])
                with self.assertRaises(ValueError):
                    factory({**BINDING, "model_id": "different@2"}).load(path)

    def test_invalid_masks_and_unfitted_jacobian_fail_closed(self):
        with self.assertRaises(RuntimeError):
            JacobianLens(lambda x: x).run(torch.ones(1, 3))
        with self.assertRaises(ValueError):
            TunedLens(lambda x: x, 2).fit(torch.ones(1, 2), torch.ones(1, 2),
                                         mask=torch.zeros(1, dtype=torch.bool))
        with self.assertRaises(ValueError):
            JacobianLens(lambda x: x).fit(torch.ones(1, 3, 2), lambda x: x)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                TunedLens(lambda x: x, 2).save(Path(directory) / "unbound.pt")


if __name__ == "__main__":
    unittest.main()
