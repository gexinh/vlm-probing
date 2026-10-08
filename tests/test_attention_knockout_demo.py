"""Coordinate/metric contracts used by the GQA Attention Knockout demo."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demos.inputs import (FixedAnswerProbability, layer_window,
                                           load_samples, pad_square, target_patch_mask)
from vlm_probing import TokenLayout


class AttentionKnockoutDemoTests(unittest.TestCase):
    def test_nine_layer_windows_clip_at_both_boundaries(self):
        self.assertEqual(layer_window(0, 32), [0, 1, 2, 3, 4])
        self.assertEqual(layer_window(15, 32), list(range(11, 20)))
        self.assertEqual(layer_window(31, 32), list(range(27, 32)))
        with self.assertRaises(ValueError):
            layer_window(3, 32, 8)

    def test_square_pad_preserves_bbox_coordinates_and_mean_fill(self):
        image = Image.new("RGB", (100, 50), (240, 10, 15))
        padded, offset = pad_square(image, [0.5, 0.4, 0.3])
        self.assertEqual(offset, (0, 25))
        self.assertEqual(padded.size, (100, 100))
        self.assertEqual(padded.getpixel((0, 0)), (127, 102, 76))
        self.assertEqual(padded.getpixel((5, 30)), image.getpixel((5, 5)))

    def test_target_region_runs_through_same_image_geometry(self):
        class ImageProcessor:
            image_mean = [0.5, 0.4, 0.3]
            def __call__(self, *, images, **kwargs):
                self.image = images
                self.settings = kwargs
                resized = images.resize((28, 28), Image.Resampling.NEAREST)
                return {"pixel_values": torch.from_numpy(np.array(resized).copy()).permute(2, 0, 1)[None]}
        transform = ImageProcessor()
        processor = SimpleNamespace(image_processor=transform)
        region, metadata = target_patch_mask(Image.new("RGB", (100, 50)), [0, 0, 30, 20], processor)
        # Top left bbox moves down by 25 pixels, remaining in only the first
        # visual patch after square padding and resizing. A raw-image scale
        # without padding would give the wrong y coordinate.
        self.assertEqual(region.tolist(), [True, False, False, False])
        self.assertEqual(metadata["padding_offset_xy"], [0, 25])
        self.assertIs(transform.settings["do_normalize"], False)
        self.assertIs(transform.settings["do_rescale"], False)

    def test_fixed_probability_uses_prompt_boundary_not_teacher_forced_answer(self):
        logits = torch.tensor([[[0., 0., 0.], [0., 2., 0.], [10., 0., 0.]]])
        layout = TokenLayout(torch.tensor([[True, True, True]]),
                             prompt=torch.tensor([[True, True, False]]))
        result = FixedAnswerProbability(1).score(logits, layout)
        self.assertAlmostEqual(result.item(), torch.softmax(logits[0, 1], 0)[1].item())

    def test_dataset_loader_checks_original_dimensions_bbox_and_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            image = path / "original.jpg"
            Image.new("RGB", (10, 6)).save(image)
            row = {"image_path": image.name, "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                   "image_size": [10, 6], "bbox_xywh": [2, 1, 3, 4]}
            manifest = path / "samples.json"
            manifest.write_text(json.dumps([row]))
            self.assertEqual(load_samples(path)[0]["bbox_xywh"], row["bbox_xywh"])
            row["bbox_xywh"] = [9, 1, 3, 4]
            manifest.write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError, "outside"):
                load_samples(path)


if __name__ == "__main__":
    unittest.main()
