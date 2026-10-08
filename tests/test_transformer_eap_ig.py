"""Native graph fidelity, directional finite differences, and IG path checks."""
import unittest

import torch

try:
    from transformers import GPT2Config, GPT2LMHeadModel
except ImportError:
    GPT2LMHeadModel = None

from vlm_probing import Prober
from vlm_probing.causal.transformer_eap_ig import TransformerEAPIG
from vlm_probing.metrics import TokenMargin


@unittest.skipIf(GPT2LMHeadModel is None, "optional transformers dependency")
class TransformerEAPIGTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(27)
        model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=8,
                                         vocab_size=23, n_positions=12,
                                         attn_pdrop=0, resid_pdrop=0, embd_pdrop=0,
                                         _attn_implementation="eager"))
        self.model, self.probe = model, Prober(model)
        self.method = TransformerEAPIG(self.probe, steps=3)
        self.clean = {"input_ids": torch.tensor([[1, 2, 3, 2]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
        self.donor = {**self.clean, "input_ids": torch.tensor([[1, 2, 3, 4]])}
        self.metric = TokenMargin(5, 6)

    def test_noop_and_full_complement_endpoints_preserve_native_model(self):
        native = self.probe._run(self.clean).logits
        trace, clean_sources, _ = self.method.graph.run(self.clean)
        torch.testing.assert_close(trace.logits, native, atol=1e-6, rtol=1e-5)
        donor_trace, donor_sources, _ = self.method.graph.run(self.donor)
        all_edges = torch.ones(len(self.method.graph.edge_names), dtype=torch.bool)
        retained, _, _ = self.method.graph.run(self.clean, donor_sources=donor_sources, keep=all_edges)
        ablated, _, _ = self.method.graph.run(self.clean, donor_sources=donor_sources, keep=~all_edges)
        torch.testing.assert_close(retained.logits, trace.logits, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(ablated.logits, donor_trace.logits, atol=1e-6, rtol=1e-5)
        self.assertEqual(len(clean_sources.shape), 4)
        self.assertEqual(len(self.method.graph.edge_names), 46)
        self.assertTrue(self.model.training)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        self.assertFalse(self.model.transformer.h[0].ln_1._forward_pre_hooks)

    def test_independent_message_gradient_matches_directional_finite_difference(self):
        graph = self.method.graph
        trace, sources, _ = graph.run(self.clean)
        _, donor_sources, _ = graph.run(self.donor)
        layout = self.probe._layout(self.clean, trace.logits)
        differences = donor_sources - sources
        scores, _ = graph.gradient_scores(self.clean, sources[:, :, 0], differences, self.metric, layout)
        # Scale the actual donor source difference by a small epsilon; all other
        # messages are unchanged, so this checks the declared physical edge.
        epsilon = 0.01
        small_donor = sources + epsilon * differences
        baseline = self.metric.score(trace.logits, layout)
        for name in ("input -> L1.H0.Q", "input -> L1.MLP"):
            edge_id = graph.edge_names.index(name)
            keep = torch.ones(len(graph.edge_names), dtype=torch.bool)
            keep[edge_id] = False
            patched, _, _ = graph.run(self.clean, donor_sources=small_donor, keep=keep)
            measured = -(self.metric.score(patched.logits, layout) - baseline).mean() / epsilon
            torch.testing.assert_close(scores[edge_id], measured, atol=2e-5, rtol=0.03)

    def test_original_input_path_and_actual_evaluation(self):
        result = self.method.run(self.clean, donor=self.donor, metric=self.metric)
        self.assertEqual(result.metadata["path"], "linear_input_embeddings")
        self.assertEqual(result.metadata["alphas"], [0, 1 / 3, 2 / 3])
        self.assertEqual(result.tensors["edge_scores"].numel(), len(self.method.graph.edge_names))
        self.assertFalse(torch.allclose(result.tensors["edge_scores"], result.tensors["eap_scores"], atol=1e-7))
        evaluation = self.method.evaluate(self.clean, donor=self.donor, metric=self.metric,
                                         attribution=result, budgets=[0, 20, len(self.method.graph.edge_names)])
        self.assertLess(float(evaluation.tensors["all_retained_error"].abs().max()), 1e-6)
        self.assertLess(float(evaluation.tensors["none_retained_error"].abs().max()), 1e-6)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        self.assertTrue(torch.isfinite(evaluation.tensors["circuit_score"]).all())
        # A requested small budget must be a true subset, not another full run.
        self.assertTrue((evaluation.tensors["retained_edges"][:, 1] <= 20).all())

    def test_graph_constraints_and_quadrature_are_explicit(self):
        graph = self.method.graph
        indices = graph.edge_indices
        self.assertEqual(len(set(graph.edge_names)), len(graph.edge_names))
        for (source, destination), name in zip(indices.tolist(), graph.edge_names):
            self.assertEqual(name, f"{graph.source_names[source]} -> {graph.destination_names[destination]}")
        with self.assertRaises(ValueError):
            TransformerEAPIG(self.probe, steps=True)
        with self.assertRaises(ValueError):
            TransformerEAPIG(self.probe, quadrature="trapezoid")
        with self.assertRaises(ValueError):
            graph.select(torch.ones(len(graph.edge_names)), -1)


if __name__ == "__main__":
    unittest.main()
