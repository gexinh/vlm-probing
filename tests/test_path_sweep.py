"""Shared-cache sweeps must equal independent four-stage path interventions."""
import unittest

import torch

from vlm_probing import Prober, TokenMargin
from test_path_patching import CircuitModel


class PathSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(194)
            self.model = CircuitModel()
        self.probe = Prober(self.model)
        self.base = {"input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])}
        self.donor = {"input_ids": torch.tensor([[5, 6, 7, 8], [8, 7, 6, 5]])}
        self.metric = TokenMargin(positive=2, negative=8)
        self.senders = [(0, 0), (0, 1), (1, 0)]

    def assert_clean(self):
        self.assertFalse(self.probe.adapter._active)
        for module in self.model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def assert_independent_runs(self, receivers):
        method = self.probe.causal.path(senders=self.senders, receivers=receivers)
        result = method.sweep(self.base, donor=self.donor, metric=self.metric, alignment="position")
        self.assertEqual(len(self.model.records), 2 + 2 * len(self.senders))
        self.assertEqual(result.metadata["actual_forward_passes"], len(self.model.records))
        self.assertEqual(result.metadata["shared_caches"], ["baseline", "donor"])
        self.assertEqual(result.metadata["sweep"], "independent sender-to-receiver paths")
        self.assertEqual(result.tensors["effect"].shape, (len(self.senders), 2))
        self.assertEqual(result.tensors["baseline_score"].shape, (2,))
        torch.testing.assert_close(result.tensors["senders"], torch.tensor(self.senders))
        self.assert_clean()
        for index, sender in enumerate(self.senders):
            single = self.probe.causal.path(senders=[sender], receivers=receivers).run(
                self.base, donor=self.donor, metric=self.metric, alignment="position")
            for name in ("baseline_score", "donor_score"):
                torch.testing.assert_close(result.tensors[name], single.tensors[name])
            for name in ("intervention_score", "effect"):
                torch.testing.assert_close(result.tensors[name][index], single.tensors[name])
        self.assert_clean()
        return result

    def test_residual_sweep_equals_independent_runs_with_two_cached_forwards(self):
        self.assert_independent_runs("residual")

    def test_qkv_receivers_equal_independent_runs_and_capture_all_receiver_sites(self):
        for receivers in ([(2, 1, "q")], [(2, 0, "k"), (2, 1, "v")],
                          [(1, 0, "q"), (2, 1, "k"), (2, 0, "v")]):
            with self.subTest(receivers=receivers):
                self.model.records.clear()
                self.assert_independent_runs(receivers)

    def test_no_sender_or_cache_mutation_leaks_between_paths(self):
        self.probe.causal.path(senders=[(0, 0), (0, 1)]).sweep(
            self.base, donor=self.donor, metric=self.metric, alignment="position")
        base, donor, first, _, second, _ = self.model.records
        torch.testing.assert_close(first["heads"][0][:, -1, 0], donor["heads"][0][:, -1, 0])
        torch.testing.assert_close(first["heads"][0][:, :, 1], base["heads"][0][:, :, 1])
        torch.testing.assert_close(second["heads"][0][:, -1, 1], donor["heads"][0][:, -1, 1])
        torch.testing.assert_close(second["heads"][0][:, :, 0], base["heads"][0][:, :, 0])
        for controlled in (first, second):
            for layer in (1, 2):
                torch.testing.assert_close(controlled["heads"][layer], base["heads"][layer])
        self.assert_clean()

    def test_scalar_metric_shapes_and_progress_order(self):
        progress = []
        result = self.probe.causal.path(senders=self.senders).sweep(
            self.base, donor=self.donor,
            metric=lambda logits: (logits[:, -1, 2] - logits[:, -1, 8]).mean(),
            alignment="position", progress=lambda *args: progress.append(args))
        self.assertEqual(result.tensors["baseline_score"].shape, ())
        self.assertEqual(result.tensors["donor_score"].shape, ())
        self.assertEqual(result.tensors["effect"].shape, (3,))
        self.assertEqual(result.metadata["score_axes"], ["path"])
        self.assertEqual(progress, [(1, 3, (0, 0)), (2, 3, (0, 1)), (3, 3, (1, 0))])
        self.assert_clean()

    def test_identity_donor_has_zero_effect_at_every_sender(self):
        result = self.probe.causal.path(senders=self.senders).sweep(
            self.base, donor=self.base, metric=self.metric)
        torch.testing.assert_close(result.tensors["effect"], torch.zeros(3, 2, dtype=torch.float64),
                                   atol=1e-12, rtol=0)
        self.assert_clean()

    def test_failures_clean_all_hooks_and_restore_training_state(self):
        self.model.train()
        self.model.blocks[1].norm1.eval()
        states = [module.training for module in self.model.modules()]
        for failed_run in range(1, 7):
            with self.subTest(failed_run=failed_run):
                self.model.records.clear()
                self.model.fail_run = failed_run
                with self.assertRaisesRegex(RuntimeError, "intentional"):
                    self.probe.causal.path(senders=self.senders[:2]).sweep(
                        self.base, donor=self.donor, metric=self.metric, alignment="position")
                self.assert_clean()
                self.assertEqual([module.training for module in self.model.modules()], states)
        self.model.fail_run = None
        self.model.records.clear()
        retry = self.probe.causal.path(senders=self.senders).sweep(
            self.base, donor=self.base, metric=self.metric)
        torch.testing.assert_close(retry.tensors["effect"], torch.zeros(3, 2, dtype=torch.float64),
                                   atol=1e-12, rtol=0)

    def test_alignment_and_metric_guards_are_preserved(self):
        method = self.probe.causal.path(senders=self.senders)
        with self.assertRaisesRegex(ValueError, "token IDs differ"):
            method.sweep(self.base, donor=self.donor, metric=self.metric)
        self.assertEqual(len(self.model.records), 2)
        self.assert_clean()
        with self.assertRaisesRegex(ValueError, "finite scalar or"):
            method.sweep(self.base, donor=self.donor, metric=lambda logits: logits.new_zeros(3),
                         alignment="position")
        self.assert_clean()

    def test_invalid_progress_is_rejected_before_model_execution(self):
        with self.assertRaisesRegex(TypeError, "progress must be callable"):
            self.probe.causal.path(senders=self.senders).sweep(
                self.base, donor=self.donor, metric=self.metric, alignment="position", progress=1)
        self.assertEqual(len(self.model.records), 0)


if __name__ == "__main__":
    unittest.main()
