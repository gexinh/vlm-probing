"""Check that presentation preserves measured cases and causal claims."""
import copy
import sys
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demos.path_display import plot_head_paths, plot_receiver_paths, prompt_cards, sample_title


class PathVisualTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_readout_ranking_uses_signed_pair_mean_not_mean_absolute_effect(self):
        figure = plot_head_paths([(0, 0), (0, 1)], [[1., -1.], [.2, .2]], top_k=1)
        text = " ".join(item.get_text() for item in figure.axes[0].texts)
        self.assertIn("Head L0.H1", text)
        self.assertNotIn("Head L0.H0", text)
        self.assertIn("independent interventions", text)

    @staticmethod
    def artifact():
        return {"edges": [{"sender": [0, 0], "receiver": [1, 0, "q"],
                           "per_pair": [{"pair_id": "a", "normalized_effect": .1},
                                        {"pair_id": "b", "normalized_effect": -.1}],
                           "mean_normalized_effect": 0.}]}

    def test_receiver_display_preserves_opposite_signs_and_zero_mean(self):
        figure = plot_receiver_paths(self.artifact())
        table = list(figure.axes[1].tables)[0]
        values = [cell.get_text().get_text() for cell in table.get_celld().values()]
        self.assertIn("+0.1000", values)
        self.assertIn("-0.1000", values)
        self.assertIn("+0.0000", values)
        text = " ".join(item.get_text() for item in figure.axes[0].texts)
        self.assertIn("Φ = 0", text)
        self.assertIn("not separately tested", text)

    def test_receiver_display_rejects_fabricated_mean_or_duplicate_route(self):
        artifact = self.artifact()
        artifact["edges"][0]["mean_normalized_effect"] = .2
        with self.assertRaisesRegex(ValueError, "mean disagrees"):
            plot_receiver_paths(artifact)
        artifact = self.artifact()
        artifact["edges"].append(copy.deepcopy(artifact["edges"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate measured edge"):
            plot_receiver_paths(artifact)

    def test_receiver_display_rejects_incomparable_case_orders(self):
        artifact = self.artifact()
        second = copy.deepcopy(artifact["edges"][0])
        second["sender"] = [0, 1]
        second["per_pair"].reverse()
        artifact["edges"].append(second)
        with self.assertRaisesRegex(ValueError, "same pair order"):
            plot_receiver_paths({"edges": [artifact["edges"][0], second]})

    def test_prompt_card_shows_full_inputs_separate_labels_and_escaped_text(self):
        pair = {"id": "appendix-hidden-source-id", "clean_prompt": "Full task <script>\nTurn left",
                "donor_prompt": "Full task <script>\nTurn right", "clean_answer": "unknown",
                "donor_answer": "avocado", "source": {"dataset": "VRUBench", "steps": 2}}
        title = sample_title(pair, 0)
        self.assertIn("unseen vs known view", title)
        self.assertIn("unknown", title)
        self.assertNotIn(pair["id"], title)
        html = prompt_cards(pair, token_metadata={"clean_margin": 2.25, "donor_margin": 3.8125}).data
        self.assertIn("Full task &lt;script&gt;", html)
        self.assertNotIn("<script>", html)
        self.assertIn("left</mark>", html)
        self.assertIn("right</mark>", html)
        self.assertIn("a scoring label, not part of the input", html)
        self.assertIn("+2.2500", html)
        self.assertIn("+3.8125", html)


if __name__ == "__main__":
    unittest.main()
