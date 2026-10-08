"""Candidate-route workflow tested with a real deterministic decoder."""
import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demos.path_routes import aggregate_pair_effects, measure_receiver_paths, select_candidates
from test_path_patching import CircuitModel
from vlm_probing import CapabilityError, Prober


class CharacterTokenizer:
    """Exact token boundaries for the real-decoder integration test."""
    chat_template = None

    def __call__(self, text, **kwargs):
        assert kwargs == {"return_tensors": "pt", "add_special_tokens": False}
        ids = torch.tensor([[ord(character) - ord("a") + 1 for character in text]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class ReceiverPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_candidate_selection_respects_decoder_order_and_physical_head_counts(self):
        counts = {layer: 2 if layer % 2 else 3 for layer in range(20)}
        ranking = [{"layer": 17, "head": 1, "mean_effect": 0.9},
                   {"layer": 19, "head": 1, "mean_effect": -0.8},
                   {"layer": 18, "head": 2, "mean_effect": 0.7},
                   {"layer": 2, "head": 2, "mean_effect": -0.6},
                   {"layer": 3, "head": 0, "mean_effect": 0.6}]
        senders, receivers = select_candidates(list(reversed(ranking)), counts, 3, 2)
        self.assertEqual(receivers, [(19, 1, "q"), (18, 2, "q")])
        self.assertEqual(senders, [(17, 1), (2, 2), (3, 0)])
        for sender in senders:
            self.assertLess(sender[0], min(receiver[0] for receiver in receivers))
        with self.assertRaisesRegex(ValueError, "unavailable"):
            select_candidates([*ranking, {"layer": 19, "head": 2, "mean_effect": 5}], counts)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_candidates([*ranking, ranking[0]], counts)
        with self.assertRaisesRegex(ValueError, "not enough ranked sender"):
            select_candidates(ranking, counts, 4, 2)

    def test_aggregation_averages_per_pair_ratios_and_keeps_zero(self):
        records = [{"raw_delta": 2, "normalizer": 4, "normalized_effect": 0.5},
                   {"raw_delta": 0, "normalizer": 1, "normalized_effect": 0.0},
                   {"raw_delta": 1, "normalizer": 0, "normalized_effect": None}]
        self.assertEqual(aggregate_pair_effects(records), 0.25)
        self.assertNotEqual(aggregate_pair_effects(records), (2 + 0) / (4 + 1))
        self.assertIsNone(aggregate_pair_effects([records[-1]]))
        with self.assertRaisesRegex(ValueError, "finite"):
            aggregate_pair_effects([{"normalized_effect": math.inf}])

    def test_workflow_records_actual_query_paths_and_zero_controls(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(194)
            model = CircuitModel()
        probe = Prober(model)
        pairs = [{"id": identifier, "clean_prompt": clean, "donor_prompt": donor,
                  "clean_answer": "c", "donor_answer": "k"}
                 for identifier, clean, donor in (("one", "abcd", "efgh"),
                                                   ("two", "dcba", "hgfe"))]
        artifact = measure_receiver_paths(probe, pairs, CharacterTokenizer(), "cpu",
                                          [(0, 0), (1, 0)], [(2, 0, "q")])
        self.assertEqual(len(artifact["edges"]), 2)
        self.assertEqual(artifact["actual_forward_passes"], 28)
        self.assertEqual(len(model.records), 28)
        self.assertEqual(len(artifact["controls"]), 4)
        self.assertTrue(all(control["passed"] for control in artifact["controls"]))
        self.assertTrue(all(abs(control["raw_delta"]) <= 1e-8 for control in artifact["controls"]))
        for edge in artifact["edges"]:
            self.assertEqual(edge["receiver"], [2, 0, "q"])
            self.assertEqual([pair["pair_id"] for pair in edge["per_pair"]], ["one", "two"])
            for pair in edge["per_pair"]:
                self.assertAlmostEqual(pair["raw_delta"], pair["patched_margin"] - pair["clean_margin"])
                self.assertAlmostEqual(pair["normalizer"], pair["donor_margin"] - pair["clean_margin"])
                self.assertAlmostEqual(pair["normalized_effect"], pair["raw_delta"] / pair["normalizer"], places=6)
            self.assertEqual(edge["mean_normalized_effect"],
                             aggregate_pair_effects(edge["per_pair"]))
        self.assertFalse(probe.adapter._active)
        for module in model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_invalid_query_capability_or_direction_fails_before_any_forward(self):
        model = CircuitModel()
        probe = Prober(model)
        pair = {"id": "one", "clean_prompt": "abcd", "donor_prompt": "efgh",
                "clean_answer": "c", "donor_answer": "k"}
        with self.assertRaisesRegex(ValueError, "precede"):
            measure_receiver_paths(probe, [pair], CharacterTokenizer(), "cpu", [(2, 0)], [(1, 0, "q")])
        probe.spec.path_qkv.pop(2)
        with self.assertRaises(CapabilityError):
            measure_receiver_paths(probe, [pair], CharacterTokenizer(), "cpu", [(0, 0)], [(2, 0, "q")])
        self.assertFalse(model.records)


if __name__ == "__main__":
    unittest.main()
