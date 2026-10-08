"""CPU semantics and strict backend checks for genuine Chefer relevance."""

import unittest

import torch

from vlm_probing.attention.chefer import CheferTransformerAttribution
from vlm_probing.attention.lrp import CheferLRP, create_chefer_vit


def tiny_model():
    torch.manual_seed(120)
    return create_chefer_vit(img_size=8, patch_size=4, embed_dim=12,
                             depth=2, num_heads=3, num_classes=3,
                             mlp_ratio=2, qkv_bias=True)


class CheferTests(unittest.TestCase):
    def test_hf_conversion_preserves_actual_logits(self):
        try:
            from transformers import ViTConfig, ViTForImageClassification
        except ImportError:
            self.skipTest("optional transformers dependency unavailable")
        torch.manual_seed(4)
        native = ViTForImageClassification(ViTConfig(
            image_size=8, patch_size=4, hidden_size=12, num_hidden_layers=2,
            num_attention_heads=3, intermediate_size=24, num_labels=3,
            layer_norm_eps=1e-12, hidden_dropout_prob=0., attention_probs_dropout_prob=0.))
        native.eval()
        copied = create_chefer_vit(source=native)
        images = torch.randn(2, 3, 8, 8)
        with torch.enable_grad():
            observed = copied(images)
        expected = native(pixel_values=images).logits
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=2e-7)
        self.assertEqual(copied.blocks[0].norm1.eps, native.config.layer_norm_eps)
        native.config.hidden_act = "gelu_new"
        with self.assertRaisesRegex(ValueError, "exact GELU"):
            create_chefer_vit(source=native)

    def test_frozen_backend_batch_equals_independent_images_and_restores_state(self):
        model = tiny_model()
        model.train()
        model.blocks[0].norm1.eval()
        states = [module.training for module in model.modules()]
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = torch.ones_like(parameter)
        images = torch.randn(2, 3, 8, 8)
        original = images.clone()
        targets = torch.tensor([0, 2])
        method = CheferLRP(model)
        result = method.run(images, target=targets)
        separate = [method.run(images[i:i + 1], target=int(targets[i])) for i in range(2)]
        torch.testing.assert_close(result.tensors["relevance"],
                                   torch.cat([value.tensors["relevance"] for value in separate]))
        self.assertEqual(result.tensors["attention_relevance"].shape, (2, 2, 3, 5, 5))
        self.assertEqual(result.tensors["patch_relevance"].shape, (2, 2, 2))
        self.assertEqual(states, [module.training for module in model.modules()])
        for parameter in model.parameters():
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
        torch.testing.assert_close(images, original)
        # LRP CAMs are propagated relevances, not forward attention probabilities.
        self.assertFalse(torch.allclose(result.tensors["attention_relevance"].sum(-1),
                                        torch.ones(2, 2, 3, 5)))

    def test_all_paper_relprop_modes_and_class_dependence(self):
        model = tiny_model()
        image = torch.randn(1, 3, 8, 8)
        first = CheferLRP(model).run(image, target=0)
        second = CheferLRP(model).run(image, target=1)
        self.assertFalse(torch.allclose(first.tensors["relevance"], second.tensors["relevance"]))
        for mode, shape in (("full", (1, 8, 8)), ("last_layer", (1, 4))):
            result = CheferLRP(model, method=mode).run(image, target=1)
            self.assertEqual(result.tensors["relevance"].shape, shape)
            self.assertTrue(torch.isfinite(result.tensors["relevance"]).all())
        top = CheferLRP(model).run(image)
        torch.testing.assert_close(top.tensors["target"], top.tensors["logits"].argmax(-1))

    def test_aggregation_uses_lrp_cams_not_probability_contract(self):
        # CAMs need not sum to one and can be signed before CAM-gradient clamping.
        cams = torch.tensor([[[[[0., -2.], [3., 0.]]]],
                             [[[[0., 1.], [-1., 0.]]]]])
        gradients = torch.tensor([[[[[1., -1.], [2., 1.]]]],
                                  [[[[1., 3.], [1., 1.]]]]])
        result = CheferTransformerAttribution().run(cams, gradients)
        expected = torch.tensor([[[19., 5.], [6., 1.]]])
        torch.testing.assert_close(result.tensors["relevance"], expected)
        self.assertFalse(result.metadata["row_normalized"])
        with self.assertRaisesRegex(ValueError, "finite"):
            CheferTransformerAttribution().run(cams, gradients * float("nan"))

    def test_rejects_unsupported_models_targets_and_inference_mode(self):
        with self.assertRaisesRegex(TypeError, "create_chefer_vit"):
            CheferLRP(torch.nn.Linear(3, 3))
        method = CheferLRP(tiny_model())
        image = torch.randn(1, 3, 8, 8)
        with self.assertRaisesRegex(ValueError, "outside"):
            method.run(image, target=3)
        with self.assertRaisesRegex(ValueError, "image size"):
            method.run(torch.randn(1, 3, 10, 10), target=0)
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, "requires autograd"):
            method.run(image, target=0)


if __name__ == "__main__":
    unittest.main()
