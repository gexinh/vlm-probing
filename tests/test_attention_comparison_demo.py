"""Check exact patch deletion and author image preprocessing without a GPU."""

import unittest

import numpy as np
from PIL import Image
import torch

from demos.inputs import delete_patches, preprocess


class AttentionComparisonDemoTests(unittest.TestCase):
    def test_deletion_uses_row_major_complete_patches_without_editing_source(self):
        original = torch.arange(1, 49, dtype=torch.float32).reshape(1, 3, 4, 4)
        saved = original.clone()
        edited = delete_patches(original, [0, 3], patch_size=2)
        expected = original.clone()
        expected[:, :, :2, :2] = 0
        expected[:, :, 2:, 2:] = 0
        torch.testing.assert_close(edited, expected)
        torch.testing.assert_close(original, saved)
        torch.testing.assert_close(delete_patches(original, [], patch_size=2), original)

    def test_complete_deletion_is_training_mean_in_normalized_coordinates(self):
        pixels = torch.randn(1, 3, 8, 8)
        result = delete_patches(pixels, list(range(16)), patch_size=2)
        self.assertEqual(result.abs().max().item(), 0)
        torch.testing.assert_close(result * .5 + .5, torch.full_like(result, .5))

    def test_author_transform_matches_explicit_bilinear_resize_and_center_crop(self):
        image = Image.fromarray(np.arange(225 * 225 * 3, dtype=np.uint8).reshape(225, 225, 3))
        crop, layout = preprocess(image)
        expected = image.resize((256, 256), Image.Resampling.BILINEAR).crop((16, 16, 240, 240))
        np.testing.assert_array_equal(np.asarray(crop), np.asarray(expected))
        self.assertEqual(layout["resize_size"], [256, 256])
        self.assertEqual(layout["crop_in_resized_image"], [16, 16, 240, 240])

    def test_invalid_patch_layout_or_index_fails(self):
        with self.assertRaises(ValueError):
            delete_patches(torch.zeros(1, 3, 5, 4), [], patch_size=2)
        with self.assertRaises(ValueError):
            delete_patches(torch.zeros(1, 3, 4, 4), [4], patch_size=2)


if __name__ == "__main__":
    unittest.main()
