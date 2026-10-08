"""Lightweight contracts for public, portable model-demo inputs."""
from pathlib import Path
from types import SimpleNamespace
import json
import unittest

import torch
from demos.ioi_inputs import BatchCandidateMargin, prepare_pairs
from vlm_probing import TokenLayout


class DemoPortabilityTests(unittest.TestCase):
    def test_batch_candidate_metric_is_constructible_and_scores_each_example(self):
        metric = BatchCandidateMargin([1, 2], [2, 0])
        logits = torch.tensor([[[0., 0., 0.], [1., 4., 2.]],
                               [[0., 0., 0.], [2., 0., 5.]]])
        layout = TokenLayout(torch.ones(2, 2, dtype=torch.bool))
        torch.testing.assert_close(metric.score(logits, layout), torch.tensor([2., 3.]))

    def test_published_pair_preparation_returns_an_executable_batch_metric(self):
        class Tokenizer:
            bos_token_id = 0
            eos_token_id = 0
            def __call__(self, text, **kwargs):
                return {"input_ids": {"one": [1, 2], "two": [2, 1],
                                      " name": [1], " subject": [2]}[text]}
        pair = {"clean_prompt": "one", "donor_prompt": "two", "positive_token": " name",
                "negative_token": " subject", "positive_id": 1, "negative_id": 2}
        clean, reference, metric = prepare_pairs([pair], Tokenizer(), "cpu")
        self.assertEqual(clean.kwargs["input_ids"].tolist(), [[0, 1, 2]])
        self.assertEqual(reference.kwargs["input_ids"].tolist(), [[0, 2, 1]])
        logits = torch.tensor([[[0., 0., 0.], [0., 0., 0.], [0., 5., 1.]]])
        torch.testing.assert_close(metric.score(logits, clean.layout), torch.tensor([4.]))

    def test_multi_piece_answer_keeps_native_target_ids_and_shifted_prediction_sites(self):
        from demos.inputs import append_answer
        from vlm_probing import ProbeInputs
        ids = torch.tensor([[3, 4, 5, 6]])
        valid = torch.ones_like(ids, dtype=torch.bool)
        visual = torch.tensor([[False, True, False, False]])
        inputs = ProbeInputs({"input_ids": ids, "attention_mask": valid.long(),
                              "position_ids": torch.arange(4)[None]},
                             TokenLayout(valid, visual, valid, ids))
        scored, answer_mask, prediction_mask = append_answer(inputs, [7, 8])
        self.assertEqual(scored.layout.token_ids[answer_mask].tolist(), [7, 8])
        self.assertEqual(prediction_mask.nonzero().tolist(), [[0, 3], [0, 4]])
        self.assertEqual(answer_mask.nonzero().tolist(), [[0, 4], [0, 5]])
        self.assertFalse((answer_mask & scored.layout.visual).any())
        self.assertEqual(scored.kwargs["input_ids"].tolist(), [[3, 4, 5, 6, 7, 8]])
        self.assertNotIn("position_ids", scored.kwargs)
        self.assertEqual(inputs.kwargs["input_ids"].shape[1], 4)


if __name__ == "__main__":
    unittest.main()
