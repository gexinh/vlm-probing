"""Dataset-level public calibration and prevention of incomplete-cache fitting."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from tiny_model import TinyModel
from vlm_probing.training import validate_cache
from vlm_probing import Prober


class LensBatchTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.model = TinyModel()
        self.probe = Prober(self.model)
        self.binding = {"model_id": "tiny-seed7", "readout_id": "norm-head",
                        "tokenizer_id": "fixture", "calibration_id": "separate-fixtures"}
        self.training = [{"input_ids": torch.tensor([[1, 2, index]]),
                          "image_tokens": torch.randn(1, 2, 6)} for index in range(3, 9)]
        self.validation = [{"input_ids": torch.tensor([[9, 8, 7]]),
                            "image_tokens": torch.randn(1, 2, 6)}]

    def test_public_fit_save_load_resume_preserves_model(self):
        original = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        for name in ("tuned", "attention"):
            factory = getattr(self.probe.lens, name)
            with tempfile.TemporaryDirectory() as directory:
                method = factory(layers=[0], binding=self.binding)
                method.fit_batches(self.training, self.validation, directory,
                                   batch_size=2, epochs=2, patience=10, seed=9)
                self.assertEqual(method.training_history[0]["epochs"][0]["optimizer_steps"], 3)
                expected = method.run(self.validation[0]).tensors["logits"]
                loaded = factory(layers=[0], binding=self.binding).load(directory)
                torch.testing.assert_close(loaded.run(self.validation[0]).tensors["logits"], expected)
                loaded.fit_batches(self.training, self.validation, directory,
                                   batch_size=2, epochs=3, patience=10, seed=9)
                self.assertEqual(loaded.training_history[0]["completed_epochs"], 3)
        self.assertTrue(self.model.training)
        for name, parameter in self.model.named_parameters():
            torch.testing.assert_close(parameter, original[name])
            self.assertIsNone(parameter.grad)

    def test_rejects_undeclared_data_identity_and_shared_iterable(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "identities"):
                self.probe.lens.tuned(layers=[0]).fit_batches(self.training, self.validation, directory)
            with self.assertRaisesRegex(ValueError, "separate"):
                self.probe.lens.tuned(layers=[0], binding=self.binding).fit_batches(
                    self.training, self.training, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_incomplete_activation_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = {"samples": [{"image_id": 1, "split": "train"},
                                    {"image_id": 2, "split": "validation"}]}
            (path / "samples.json").write_text(json.dumps(manifest))
            identity = hashlib.sha256((path / "samples.json").read_bytes()).hexdigest()
            (path / "config.json").write_text(json.dumps({"manifest_sha256": identity, "layers": [7]}))
            (path / "status.json").write_text(json.dumps({"state": "extracting"}))
            with self.assertRaisesRegex(ValueError, "finish"):
                validate_cache(path, path, [7])
            (path / "status.json").write_text(json.dumps({"state": "complete", "complete_images": {
                "train": 1, "validation": 1}}))
            with self.assertRaisesRegex(ValueError, "complete manifest"):
                validate_cache(path, path, [7])
            for split, image in (("train", 1), ("validation", 2)):
                torch.save({"image_ids": [image], "positions": [[image, 8]],
                    "teacher_logits": torch.zeros(1, 3), "activations": {7: torch.zeros(1, 2)}},
                    path / f"{split}_0000.pt")
            _, counts = validate_cache(path, path, [7])
            self.assertEqual(counts["train"]["prediction_positions"], 1)
