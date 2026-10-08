"""Calibration workflow controls: exact grouped gradients and Adam resume."""
import tempfile
import unittest
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from vlm_probing.lenses import JacobianLens, TunedLens
from vlm_probing.training import train_distribution_lens
from vlm_probing.training.jacobian import jacobian_sums_from_graph

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from tiny_model import TinyModel
from vlm_probing import Prober
from vlm_probing.training import fit_jacobian_lenses


BINDING = {"model_id": "toy@1", "site": "residual.0", "readout_id": "toy-head@1",
           "tokenizer_id": "toy@1", "calibration_id": "disjoint-toy@1"}


class LensTrainingTests(unittest.TestCase):
    def test_streaming_jacobian_preserves_modes_and_equal_prompt_weighting(self):
        torch.manual_seed(31)
        model = TinyModel()
        model.norm.eval()
        model.layers[0].mlp[0].weight.requires_grad_(False)
        modes = [module.training for module in model.modules()]
        flags = [parameter.requires_grad for parameter in model.parameters()]
        probe = Prober(model)
        inputs = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                  "image_tokens": torch.randn(3, 2, 6)}
        batches = [{key: value[:2] for key, value in inputs.items()},
                   {key: value[2:] for key, value in inputs.items()}]
        with tempfile.TemporaryDirectory() as directory:
            fitted, report = fit_jacobian_lenses(probe, batches, layers=[0, 1], binding=BINDING,
                directory=directory, dim_batch=2, skip_first=0, exclude_last=False,
                tokens="all", checkpoint_every=1)
            reference = probe.lens.jacobian(layers=[0, 1], tokens="all", binding=BINDING)
            reference.fit(inputs, skip_first=0, exclude_last=False)
            self.assertEqual(report["prompts"], 3)
            for layer in [0, 1]:
                torch.testing.assert_close(fitted.kernels[layer].jacobian, reference.kernels[layer].jacobian)
            torch.testing.assert_close(fitted.kernels[1].jacobian, torch.eye(6))
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertEqual(flags, [parameter.requires_grad for parameter in model.parameters()])
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_initial_readout_is_retained_when_training_does_not_improve_it(self):
        torch.manual_seed(29)
        head = nn.Linear(2, 4)
        inputs = torch.randn(8, 2)
        rows = [{"activations": x, "teacher_logits": head(x).detach()} for x in inputs]
        lens = TunedLens(head, 2, binding=BINDING)
        with tempfile.TemporaryDirectory() as directory:
            result = train_distribution_lens(lens, DataLoader(rows, batch_size=4),
                DataLoader(rows, batch_size=4), directory, epochs=2, lr=10.0, patience=1)
            self.assertEqual(result["best_epoch"], 0)
            torch.testing.assert_close(lens.translator.weight, torch.zeros_like(lens.translator.weight))
            torch.testing.assert_close(lens.translator.bias, torch.zeros_like(lens.translator.bias))
            self.assertFalse(lens.is_fitted)

    def test_grouped_vjps_match_exact_reference_at_multiple_sources(self):
        torch.manual_seed(11)
        values = torch.randn(2, 6, 3)
        first, second = torch.randn(3, 3), torch.randn(3, 3)
        mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
        leaf = values.clone().requires_grad_(True)
        middle = leaf.cumsum(1) @ first.T
        final = middle.cumsum(1) @ second.T
        sums = jacobian_sums_from_graph(final, {0: leaf, 1: middle}, mask, mask, dim_batch=2)
        reference0 = JacobianLens(lambda x: x).fit(values,
            lambda x: (x.cumsum(1) @ first.T).cumsum(1) @ second.T,
            valid_mask=mask, skip_first=0, exclude_last=False)
        reference1 = JacobianLens(lambda x: x).fit(middle.detach(),
            lambda x: x.cumsum(1) @ second.T,
            valid_mask=mask, skip_first=0, exclude_last=False)
        torch.testing.assert_close((sums[0] / 2).float(), reference0.jacobian)
        torch.testing.assert_close((sums[1] / 2).float(), reference1.jacobian)

    def test_persistent_adam_resume_matches_uninterrupted_training(self):
        torch.manual_seed(13)
        head = nn.Linear(2, 4)
        before = {name: parameter.detach().clone() for name, parameter in head.named_parameters()}
        inputs = torch.randn(16, 2)
        teacher = head(2 * inputs + torch.tensor([0.5, -0.2])).detach()
        rows = [{"activations": x, "teacher_logits": y} for x, y in zip(inputs, teacher)]

        def loader(shuffle):
            return DataLoader(rows, batch_size=4, shuffle=shuffle,
                generator=torch.Generator().manual_seed(17))

        with tempfile.TemporaryDirectory() as directory:
            full = TunedLens(head, 2, binding=BINDING)
            resumed = TunedLens(head, 2, binding=BINDING)
            train_distribution_lens(full, loader(True), loader(False), Path(directory) / "full",
                epochs=6, lr=0.02, patience=20, seed=19)
            train_distribution_lens(resumed, loader(True), loader(False), Path(directory) / "resumed",
                epochs=3, lr=0.02, patience=20, seed=19)
            final = train_distribution_lens(resumed, loader(True), loader(False), Path(directory) / "resumed",
                epochs=6, lr=0.02, patience=20, seed=19)
            for key, value in full.translator.state_dict().items():
                torch.testing.assert_close(resumed.translator.state_dict()[key], value)
            self.assertLess(final["best_validation_kl"], final["initial_validation"]["kl"])
            self.assertEqual(len(final["epochs"]), 6)
            state = torch.load(Path(directory) / "resumed/training_state.pt", weights_only=True)
            self.assertEqual(int(state["optimizer"]["state"][0]["step"]), 24)
            with self.assertRaisesRegex(ValueError, "does not match"):
                train_distribution_lens(resumed, loader(True), loader(False), Path(directory) / "resumed",
                    epochs=7, lr=0.03, patience=20, seed=19)
        for name, parameter in head.named_parameters():
            torch.testing.assert_close(parameter, before[name])
            self.assertIsNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()
