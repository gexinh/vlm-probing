"""Plotting tests verify measured values, native coordinates, and scale contracts."""
import sys
import subprocess
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vlm_probing import ProbeResult, Prober, visualization as viz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from tiny_model import TinyModel


class VisualizationTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_lens_result_preserves_native_layer_and_packed_coordinates(self):
        logits = torch.tensor([[[0., 4., 1.], [3., 2., 1.], [1., 0., 5.]],
                               [[0., 1., 4.], [0., 4., 1.], [2., 3., 1.]]])
        result = ProbeResult("logit_lens", {"logits": logits,
            "positions": torch.tensor([[0, 3], [1, 7], [0, 9]])}, {"layers": [2, 8]})
        ax = viz.plot_lens_heatmap(result, target_token_id=1, colorbar=False)
        np.testing.assert_array_equal(ax.images[0].get_array(), [[1, 3], [2, 1]])
        self.assertEqual([x.get_text() for x in ax.get_yticklabels()], ["2", "8"])
        self.assertEqual([x.get_text() for x in ax.get_xticklabels()], ["3", "9"])
        self.assertEqual(ax.images[0].norm.vmin, 1)
        self.assertEqual(ax.images[0].norm.vmax, 3)
        torch.testing.assert_close(result.tensors["logits"], logits)

    def test_lens_probabilities_use_full_vocabulary_softmax(self):
        logits = np.array([[[1., 2., 4.], [0., 1., 3.]]])
        ax = viz.plot_lens_heatmap(logits, target_token_id=[1, 2], value_kind="probability",
                                  colorbar=False)
        probabilities = torch.as_tensor(logits).softmax(-1)
        np.testing.assert_allclose(ax.images[0].get_array(), [[probabilities[0, 0, 1], probabilities[0, 1, 2]]])

    def test_attention_lens_requires_explicit_head(self):
        logits = torch.tensor([[[[1., 2.], [3., 0.]]]])
        with self.assertRaisesRegex(ValueError, "head index"):
            viz.plot_lens_heatmap(logits, target_token_id=1)
        ax = viz.plot_lens_heatmap(logits, target_token_id=1, head=1, colorbar=False)
        np.testing.assert_array_equal(ax.images[0].get_array(), [[2]])

    def test_native_decoded_text_is_not_translated_or_filtered(self):
        class Decoder:
            def decode(self, indices):
                return "中文" if indices == [1] else "word"
        ax = viz.plot_lens_heatmap(np.array([[[0., 1.]]]), target_token_id=1,
                                  tokenizer=Decoder(), colorbar=False)
        self.assertEqual(ax.texts[0].get_text(), "中文")

    def test_lens_matrix_validation_and_missing_cells(self):
        ax = viz.plot_lens_heatmap([[1, np.nan], [4, 2]], layer_labels=[3, 5],
                                  token_labels=["a", "b"], colorbar=False)
        self.assertTrue(ax.images[0].get_array().mask[0, 1])
        for kwargs in ({"layer_labels": [1, 2]}, {"token_labels": [1]},
                       {"annotations": [["x"]]}):
            with self.assertRaises(ValueError):
                viz.plot_lens_heatmap([[1, 2]], **kwargs)
        with self.assertRaisesRegex(ValueError, "positive"):
            viz.plot_lens_heatmap([[0, 1]])

    def test_probe_lens_integration_on_actual_forward(self):
        torch.manual_seed(10)
        probe = Prober(TinyModel())
        result = probe.lens.logit(layers=[0, 1], tokens="all").run({
            "input_ids": torch.tensor([[1, 2, 3]]), "image_tokens": torch.randn(1, 2, 6)})
        ax = viz.plot_lens_heatmap(result, target_token_id=4, colorbar=False)
        expected = 1 + (result.tensors["logits"] > result.tensors["logits"][..., 4, None]).sum(-1)
        np.testing.assert_array_equal(ax.images[0].get_array(), expected)

    def test_overlay_requires_grid_and_keeps_image_geometry_and_data(self):
        image = np.zeros((20, 40, 3), dtype=np.uint8)
        values = np.array([0., .2, .5, 1.])
        with self.assertRaisesRegex(ValueError, "grid_shape"):
            viz.plot_attention_overlay(image, values)
        ax = viz.plot_attention_overlay(image, values, grid_shape=(2, 2), colorbar=False)
        np.testing.assert_array_equal(ax.images[1].get_array(), [[0, .2], [.5, 1]])
        self.assertEqual(tuple(ax.images[1].get_extent()), (-.5, 39.5, 19.5, -.5))
        np.testing.assert_array_equal(values, [0, .2, .5, 1])
        np.testing.assert_array_equal(image, np.zeros((20, 40, 3), dtype=np.uint8))

    def test_overlay_selects_query_key_coordinates_in_declared_order(self):
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        values = torch.arange(25).reshape(1, 5, 5).float()
        result = ProbeResult("attention_rollout", {"rollout": values})
        ax = viz.plot_attention_overlay(image, result, visual_positions=[4, 1],
                                        query_positions=[-1], grid_shape=(1, 2), colorbar=False)
        np.testing.assert_array_equal(ax.images[1].get_array(), [[24, 21]])
        with self.assertRaisesRegex(ValueError, "both"):
            viz.plot_attention_overlay(image, result, visual_positions=[1, 4], grid_shape=(1, 2))

    def test_raw_attention_head_reduction_and_query_mask(self):
        attentions = torch.tensor([[[[[0., 1., 3.], [1., 2., 4.], [3., 5., 7.]],
                                      [[2., 3., 5.], [3., 4., 6.], [5., 7., 9.]]]]])
        result = ProbeResult("attention_profile", {"attention": attentions,
                             "query_mask": torch.tensor([[False, True, False]])})
        ax = viz.plot_attention_overlay(np.zeros((5, 5, 3), np.uint8), result,
                                        visual_positions=[0, 2], grid_shape=(1, 2), colorbar=False)
        np.testing.assert_array_equal(ax.images[1].get_array(), [[2, 5]])

    def test_grad_cam_and_attattr_default_tensor_fields_integrate(self):
        from vlm_probing.attention import AttentionAttribution, AttentionGradCAM
        attention = torch.tensor([[[[.1, .3, .6]]]])
        gradients = torch.tensor([[[[-.1, .4, .2]]]])
        image = np.zeros((5, 10, 3), np.uint8)
        cam = AttentionGradCAM().run(attention, gradients)
        ax = viz.plot_attention_overlay(image, cam, query_positions=[0],
                                        visual_positions=[1, 2], grid_shape=(1, 2), colorbar=False)
        np.testing.assert_allclose(ax.images[1].get_array(), cam.tensors["cam"][0, :, 1:])
        attattr = AttentionAttribution().run(attention, gradients)
        ax = viz.plot_attention_overlay(image, attattr, query_positions=[0],
                                        visual_positions=[0, 1, 2], grid_shape=(1, 3),
                                        signed=True, colorbar=False)
        np.testing.assert_allclose(ax.images[1].get_array(), attattr.tensors["token_attribution"][0])

    def test_batched_chefer_patch_relevance_is_spatial(self):
        result = ProbeResult("chefer_transformer_attribution", {
            "relevance": torch.arange(4).reshape(1, 4),
            "patch_relevance": torch.arange(4).reshape(1, 2, 2)})
        ax = viz.plot_attention_overlay(np.zeros((4, 8, 3), np.uint8), result,
                                        grid_shape=(2, 2), colorbar=False)
        np.testing.assert_array_equal(ax.images[1].get_array(), [[0, 1], [2, 3]])

    def test_signed_attention_keeps_negative_values_and_shared_scale(self):
        figure, axes = viz.plot_attention_comparison(np.zeros((10, 20, 3), np.uint8),
            {"a": np.array([-2., 1.]), "b": np.array([-.1, .2])}, grid_shape=(1, 2),
            signed=True, shared_scale=True)
        self.assertEqual(len(figure.axes), 3)  # two panels, one shared colorbar
        self.assertIs(axes[0, 0].images[1].norm, axes[0, 1].images[1].norm)
        norm = axes[0, 0].images[1].norm
        self.assertEqual((norm.vmin, norm.vcenter, norm.vmax), (-2, 0, 2))
        np.testing.assert_array_equal(axes[0, 0].images[1].get_array(), [[-2, 1]])
        with self.assertRaisesRegex(ValueError, "negative"):
            viz.plot_attention_overlay(np.zeros((5, 5)), [[-1, 1]], signed=False)

    def test_mixed_comparison_has_one_scale_per_signedness_group(self):
        figure, _ = viz.plot_attention_comparison(np.zeros((5, 5, 3), np.uint8),
            {"raw": [[0, 1]], "attattr": [[-3, 1]], "other": [[0, .1]]},
            signed={"raw": False, "attattr": True, "other": False}, shared_scale=True)
        self.assertEqual(len(figure.axes), 5)  # three panels, two colorbars

    def test_coordinate_grid_keeps_unmeasured_sites_distinct_from_zero(self):
        grid, rows, columns = viz.coordinate_grid([[4, 3], [1, 2]], [0, -.5],
            row_labels=[1, 2, 4], column_labels=[2, 3])
        np.testing.assert_array_equal(grid, [[-.5, np.nan], [np.nan, np.nan], [np.nan, 0]])
        self.assertEqual((rows, columns), ([1, 2, 4], [2, 3]))
        with self.assertRaisesRegex(ValueError, "unique"):
            viz.coordinate_grid([[1, 2], [1, 2]], [0, 1])

    def test_path_result_grid_and_highlights_use_actual_sender_coordinates(self):
        result = ProbeResult("path_patching", {"senders": torch.tensor([[8, 2], [3, 4], [8, 4]]),
            "effect": torch.tensor([[-.2], [.1], [0.]])})
        ax = viz.plot_causal_heatmap(result, highlight_count=1, colorbar=False)
        np.testing.assert_allclose(ax.images[0].get_array(), [[np.nan, .1], [-.2, 0]])
        self.assertEqual([t.get_text() for t in ax.get_yticklabels()], ["3", "8"])
        self.assertEqual(ax.patches[0].get_xy(), (-.5, .5))
        self.assertEqual(ax.images[0].norm.vcenter, 0)

    def test_token_scores_and_effects_have_different_column_contracts(self):
        result = ProbeResult("attribution_patching", {"token_scores": torch.arange(12).reshape(2, 2, 3),
                "estimated_effect": torch.tensor([1., 2.])}, {"layers": [0, 7]})
        ax = viz.plot_causal_heatmap(result, tensor="token_scores", batch_index=1, colorbar=False)
        np.testing.assert_array_equal(ax.images[0].get_array(), [[3, 4, 5], [9, 10, 11]])
        effect = ProbeResult("activation_patching", {"effect": torch.tensor([[1, 2], [3, 4]])}, {"layers": [1, 4]})
        ax = viz.plot_causal_heatmap(effect, batch_index=1, colorbar=False)
        np.testing.assert_array_equal(ax.images[0].get_array(), [[2], [4]])

    def test_curves_preserve_native_x_and_compute_relative_change(self):
        result = ProbeResult("attention_knockout", {"intervention_score": torch.tensor([[.4], [.2]])},
                             {"layers": [3, 8]})
        ax = viz.plot_intervention_curves({"route": result}, baseline=.4, relative=True)
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [3, 8])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [0, -.5], atol=1e-7)
        with self.assertRaisesRegex(ValueError, "nonzero"):
            viz.plot_intervention_curves({"route": [1, 2]}, relative=True, baseline=0)
        with self.assertRaisesRegex(ValueError, "match"):
            viz.plot_intervention_curves({"route": [1, 2]}, x=[1])

    def test_connections_draw_only_supplied_edges(self):
        ax = viz.plot_connections([("input", "h1", -.1), ("h1", "output", .5),
                                   ("input", "output", .2)],
                                  node_layers={"input": -1, "h1": 2, "output": 4},
                                  top_k=2, colorbar=False)
        self.assertEqual(len(ax.patches), 2)
        self.assertEqual(len(ax.collections), 3)
        with self.assertRaisesRegex(ValueError, "endpoints"):
            viz.plot_connections([("missing", "output", .1)], node_layers={"output": 4})

    def test_input_boxes_use_image_coordinates_and_keep_source_image(self):
        from PIL import Image
        original = Image.new("RGB", (40, 20), (70, 80, 90))
        before = original.tobytes()
        ax = viz.plot_input(original, prompt="What color is the shirt?", answer="Yellow",
                            bounding_boxes=[(5, 3, 8, 6)])
        self.assertEqual(ax.patches[0].get_xy(), (5, 3))
        self.assertEqual((ax.patches[0].get_width(), ax.patches[0].get_height()), (8, 6))
        self.assertIn("Answer: Yellow", ax.get_title())
        self.assertEqual(original.tobytes(), before)
        with self.assertRaises(ValueError):
            viz.plot_input(original, bounding_boxes=[(1, 1, float("nan"), 2)])

    def test_core_import_does_not_require_plotting_dependencies(self):
        script = '''
import importlib.abc
import sys
class MissingOptionalPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'matplotlib', 'PIL'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, MissingOptionalPackages())
import vlm_probing
assert not any(name in sys.modules for name in ('numpy', 'matplotlib', 'PIL'))
assert callable(vlm_probing.visualization.plot_lens_heatmap)
'''
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
