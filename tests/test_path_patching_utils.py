"""Notebook helpers checked without model downloads or GPU execution."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import json
import sys
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demos.path_helpers import (effect_grid, first_answer_metric, head_grid,
                                 load_model, load_pairs, normalized_effect, plot_heatmaps,
                                 prepare_inputs)
from vlm_probing import CapabilityError, ProbeResult, TokenLayout, TokenMargin


def fake_probe(heads):
    points = {layer: SimpleNamespace(heads=count) for layer, count in heads.items()}
    return SimpleNamespace(spec=SimpleNamespace(path_heads=points), describe=lambda: {
        "methods": {"causal.path": {"available": bool(points), "missing": "head outputs"}}})


class Tokenizer:
    def __init__(self):
        self.chat_template = "native chat format"
        self.messages, self.tokenizer_kwargs = [], []
        self.texts = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(copy.deepcopy(messages))
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return messages[-1]["content"] + "<assistant>"

    def __call__(self, text, **kwargs):
        self.texts.append(text)
        self.tokenizer_kwargs.append(kwargs)
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3).long()}

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"answer": [1, 4], "other": [2], "also": [1, 5]}[text]


class Processor:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.chat_template = "native multimodal chat format"
        self.messages, self.images = [], []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(copy.deepcopy(messages))
        assert isinstance(messages[0]["content"], list)
        assert isinstance(messages[-1]["content"], list)
        image = "<image>" if any(part["type"] == "image" for part in messages[-1]["content"]) else ""
        return image + messages[-1]["content"][-1]["text"] + "<assistant>"

    def __call__(self, *, text, images, return_tensors):
        self.images.extend(images)
        return {"input_ids": torch.tensor([[1, 9, 3]]), "pixel_values": torch.ones(1, 3, 2, 2)}


class ContextTokenizer(Tokenizer):
    """Mimic standalone leading-space tokens and in-context suffix tokens."""

    def __init__(self):
        super().__init__()
        self.encodings = {"answer": [1, 4], "other": [2], "clean:": [7, 8], "donor:": [9, 8],
                          "clean:answer": [7, 8, 3, 4], "clean:other": [7, 8, 2],
                          "donor:answer": [9, 8, 3, 4], "donor:other": [9, 8, 2]}

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return self.encodings[text]

    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return "clean:" if ids == [7, 8] else "donor:"


class PathPatchingUtilsTests(unittest.TestCase):
    def test_load_pair_contract_and_relative_images(self):
        pair = {"id": "example", "clean_prompt": "clean", "donor_prompt": "donor",
                "clean_answer": "answer", "donor_answer": "other", "clean_image": "images/a.png"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            path.write_text(json.dumps(pair) + "\n", encoding="utf-8")
            loaded = load_pairs(path)[0]
            self.assertEqual(loaded["clean_image"], str(Path(directory) / "images/a.png"))
            path.write_text(json.dumps(pair) + "\n" + json.dumps(pair), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate pair id"):
                load_pairs(path)
            for field, value in (("id", ""), ("donor_answer", 3), ("source", [])):
                path.write_text(json.dumps({**pair, field: value}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_pairs(path)
            path.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                load_pairs(path)

    def test_published_pairs_preserve_author_sources_and_answer_flip(self):
        path = Path(__file__).resolve().parents[1] / "demos/assets/vrubench_paper_samples.jsonl"
        pairs = load_pairs(path)
        self.assertEqual(len(pairs), 2)
        for pair in pairs:
            self.assertEqual(pair["source"]["kind"], "author_published_paper_example")
            self.assertFalse(pair["source"]["official_dataset_release"])
            clean, donor = pair["clean_prompt"].splitlines(), pair["donor_prompt"].splitlines()
            self.assertEqual(clean[:-2], donor[:-2])
            self.assertNotEqual(clean[-2], donor[-2])
            self.assertEqual(clean[-1], "Observation:")

    def test_prepare_text_inputs_uses_tokenizer_even_for_vlm(self):
        pair = {"clean_prompt": "clean", "donor_prompt": "donor",
                "source": {"system_prompt": "paper system"}}
        for processor in (Tokenizer(), Processor()):
            before = copy.deepcopy(pair)
            clean, donor = prepare_inputs(pair, processor, "cpu")
            tokenizer = getattr(processor, "tokenizer", processor)
            if isinstance(processor, Processor):
                self.assertEqual(processor.messages[0][0]["content"],
                                 [{"type": "text", "text": "paper system"}])
                self.assertEqual(processor.messages[0][-1]["content"],
                                 [{"type": "text", "text": "clean"}])
                self.assertEqual(processor.messages[1][-1]["content"],
                                 [{"type": "text", "text": "donor"}])
                self.assertFalse(tokenizer.messages)
            else:
                self.assertEqual(tokenizer.messages[0][0]["content"], "paper system")
                self.assertEqual(tokenizer.messages[0][-1]["content"], "clean")
                self.assertEqual(tokenizer.messages[1][-1]["content"], "donor")
            self.assertEqual(tokenizer.tokenizer_kwargs[0],
                             {"return_tensors": "pt", "add_special_tokens": False})
            self.assertEqual(set(clean), {"input_ids", "attention_mask"})
            self.assertEqual(set(donor), set(clean))
            self.assertEqual(pair, before)
            if isinstance(processor, Processor):
                self.assertFalse(processor.images)

    def test_vlm_processor_only_chat_template_keeps_text_only_user_prompt(self):
        processor = Processor()
        processor.tokenizer.chat_template = None
        pair = {"clean_prompt": "LLaVA clean task", "donor_prompt": "LLaVA donor task"}
        prepare_inputs(pair, processor, "cpu")
        self.assertEqual(processor.tokenizer.texts,
                         ["LLaVA clean task<assistant>", "LLaVA donor task<assistant>"])
        self.assertFalse(processor.images)
        self.assertFalse(processor.tokenizer.messages)

    def test_chat_template_that_silently_drops_user_prompt_is_rejected(self):
        processor = Processor()
        processor.apply_chat_template = lambda *args, **kwargs: "<assistant>"
        with self.assertRaisesRegex(ValueError, "omitted the user prompt"):
            prepare_inputs({"clean_prompt": "important task", "donor_prompt": "other task"},
                           processor, "cpu")

    def test_processor_selects_its_native_default_named_chat_template(self):
        processor = Processor()
        processor.chat_template = {"default": "native format", "tools": "other native format"}
        original = processor.apply_chat_template

        def render(messages, **kwargs):
            self.assertNotIn("chat_template", kwargs)
            return original(messages, **kwargs)

        processor.apply_chat_template = render
        prepare_inputs({"clean_prompt": "clean", "donor_prompt": "donor"}, processor, "cpu")

    def test_prepare_image_inputs_keep_same_image_for_text_counterfactual(self):
        from PIL import Image
        processor = Processor()
        image = Image.new("RGB", (2, 2))
        pair = {"clean_prompt": "clean", "donor_prompt": "donor", "clean_image": image}
        clean, donor = prepare_inputs(pair, processor, "cpu")
        self.assertEqual(set(clean), {"input_ids", "pixel_values"})
        self.assertEqual(set(donor), set(clean))
        self.assertIs(processor.images[0], image)
        self.assertIs(processor.images[1], image)
        self.assertEqual(processor.messages[0][-1]["content"][0], {"type": "image"})
        with self.assertRaisesRegex(ValueError, "multimodal processor"):
            prepare_inputs(pair, Tokenizer(), "cpu")

    def test_base_llm_without_chat_template_receives_unmodified_prompt(self):
        tokenizer = Tokenizer()
        tokenizer.chat_template = None
        pair = {"clean_prompt": "raw clean", "donor_prompt": "raw donor",
                "source": {"system_prompt": "do not insert this"}}
        prepare_inputs(pair, tokenizer, "cpu")
        self.assertEqual(tokenizer.texts, ["raw clean", "raw donor"])
        self.assertFalse(tokenizer.messages)

    def test_model_loader_dispatches_native_llm_and_vlm_without_remote_code(self):
        module = SimpleNamespace(**{name: MagicMock() for name in (
            "AutoConfig", "AutoModelForCausalLM", "AutoModelForImageTextToText",
            "AutoProcessor", "AutoTokenizer")})
        for is_vlm in (False, True):
            module.AutoConfig.from_pretrained.return_value = SimpleNamespace(
                vision_config={} if is_vlm else None)
            with patch.dict(sys.modules, {"transformers": module}):
                model, processor = load_model("native-checkpoint", device="cpu")
            model_class = module.AutoModelForImageTextToText if is_vlm else module.AutoModelForCausalLM
            processor_class = module.AutoProcessor if is_vlm else module.AutoTokenizer
            model_class.from_pretrained.assert_called_once_with(
                "native-checkpoint", dtype=torch.float32, attn_implementation="sdpa", device_map="cpu")
            self.assertIs(model, model_class.from_pretrained.return_value.eval.return_value)
            self.assertIs(processor, processor_class.from_pretrained.return_value)

    def test_first_answer_margin_uses_layout_and_float32(self):
        metric, information = first_answer_metric(
            {"clean_answer": "answer", "donor_answer": "other"}, Tokenizer())
        logits = torch.zeros(1, 3, 5, dtype=torch.bfloat16)
        logits[0, 1, 1], logits[0, 1, 2] = 5, 2
        logits[0, 2, 1] = 100  # Padding cannot affect the prediction score.
        layout = TokenLayout(valid=torch.tensor([[True, True, False]]),
                             token_ids=torch.tensor([[1, 2, 0]]))
        score = metric.score(logits, layout)
        self.assertEqual(score.tolist(), [3])
        self.assertEqual(score.dtype, torch.float32)
        self.assertEqual(information["clean_token_ids"], [1, 4])
        self.assertEqual((information["positive"], information["negative"]), (1, 2))
        # Subtracting bfloat16 candidates first would round away this difference.
        logits[0, 1, 1], logits[0, 1, 2] = 7.0625, .015625
        oracle = TokenMargin(metric.positive, metric.negative).score(logits.float(), layout)
        torch.testing.assert_close(metric.score(logits, layout), oracle, rtol=0, atol=0)
        self.assertNotEqual(oracle.item(), (logits[0, 1, 1] - logits[0, 1, 2]).float().item())
        metric.positive = -1
        with self.assertRaisesRegex(ValueError, "outside the vocabulary"):
            metric.score(logits, layout)
        with self.assertRaisesRegex(ValueError, "share their first token"):
            first_answer_metric({"clean_answer": "answer", "donor_answer": "also"}, Tokenizer())

    def test_context_scoring_derives_actual_continuation_not_standalone_space_token(self):
        tokenizer = ContextTokenizer()
        pair = {"clean_answer": "answer", "donor_answer": "other"}
        clean = {"input_ids": torch.tensor([[7, 8, 0]]), "attention_mask": torch.tensor([[1, 1, 0]])}
        donor = {"input_ids": torch.tensor([[9, 8]])}
        metric, information = first_answer_metric(pair, tokenizer, clean_inputs=clean, donor_inputs=donor)
        self.assertEqual((metric.positive, metric.negative), (3, 2))
        self.assertEqual(information["clean_token_ids"], [3, 4])
        self.assertEqual(information["candidate_tokenization"], "context_checked")
        self.assertNotEqual(information["positive"], tokenizer.encode("answer", False)[0])

    def test_context_scoring_rejects_merges_non_roundtrips_and_candidate_changes(self):
        pair = {"clean_answer": "answer", "donor_answer": "other"}
        clean = {"input_ids": torch.tensor([[7, 8]])}
        donor = {"input_ids": torch.tensor([[9, 8]])}
        for encoding, ids, error in (
            ("clean:answer", [7, 6, 3, 4], "prompt token boundary"),
            ("clean:", [7, 88], "do not round-trip"),
            ("donor:answer", [9, 8, 5, 4], "differ across"),
        ):
            tokenizer = ContextTokenizer()
            tokenizer.encodings[encoding] = ids
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                first_answer_metric(pair, tokenizer, clean_inputs=clean, donor_inputs=donor)
        with self.assertRaisesRegex(ValueError, "provide both"):
            first_answer_metric(pair, ContextTokenizer(), clean_inputs=clean)

    def test_prepared_image_context_reuses_native_bos_and_image_expansion(self):
        from PIL import Image

        class NativeImageProcessor(Processor):
            def __call__(self, *, text, images, return_tensors):
                # Native processing inserts BOS and expands the image token.
                ids = [1, 9, 9, 9, 8]
                if text[0].endswith("answer"):
                    ids += [3, 4]
                elif text[0].endswith("other"):
                    ids += [2]
                return {"input_ids": torch.tensor([ids]), "pixel_values": torch.ones(1, 3, 2, 2)}

        processor = NativeImageProcessor()
        pair = {"clean_prompt": "clean", "donor_prompt": "donor",
                "clean_answer": "answer", "donor_answer": "other",
                "clean_image": Image.new("RGB", (2, 2))}
        clean, donor = prepare_inputs(pair, processor, "cpu")
        self.assertEqual(set(clean), {"input_ids", "pixel_values"})
        self.assertNotIn("encode_continuation", clean)
        metric, information = first_answer_metric(pair, processor.tokenizer,
                                                  clean_inputs=clean, donor_inputs=donor)
        self.assertEqual((metric.positive, metric.negative), (3, 2))
        self.assertEqual(information["context_encoders"],
                         ["native_input_encoder", "native_input_encoder"])

    def test_normalization_per_pair_no_clipping_and_near_zero_exclusion(self):
        result = ProbeResult("path", {
            "baseline_score": torch.tensor([0., 0., 0.]),
            "donor_score": torch.tensor([2., 4., 1e-8]),
            "intervention_score": torch.tensor([[4., 2., 3.], [-2., 8., 4.]]),
        })
        values, eligible = normalized_effect(result)
        self.assertEqual(eligible.tolist(), [True, True, False])
        torch.testing.assert_close(values[:, :2], torch.tensor([[2., .5], [-1., 2.]]))
        self.assertTrue(torch.isnan(values[:, 2]).all())
        self.assertEqual(values[0].nanmean().item(), 1.25)
        self.assertNotEqual(values[0].nanmean().item(), (4. + 2.) / (2. + 4.))
        for epsilon in (0, -1, float("inf")):
            with self.assertRaises(ValueError):
                normalized_effect(result, epsilon)
        result.tensors["intervention_score"][0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            normalized_effect(result)

    def test_grid_discovers_nonuniform_heads_and_keeps_missing_cells_nan(self):
        probe = fake_probe({0: 2, 2: 3})
        self.assertEqual(head_grid(probe), [(0, 0), (0, 1), (2, 0), (2, 1), (2, 2)])
        grid = effect_grid(probe, [(0, 1), (2, 2)], torch.tensor([.5, -.4]))
        self.assertEqual(grid.shape, (3, 3))
        self.assertEqual(grid[0, 1].item(), .5)
        self.assertTrue(torch.isnan(grid[1]).all())
        self.assertTrue(torch.isnan(grid[0, 2]))
        with self.assertRaisesRegex(ValueError, "unavailable"):
            effect_grid(probe, [(0, 2)], [1.])
        with self.assertRaises(CapabilityError):
            head_grid(fake_probe({}))

    def test_plot_mean_is_mean_of_normalized_pair_values(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        probe = fake_probe({0: 2, 1: 1})
        senders = head_grid(probe)
        effects = torch.tensor([[2., .5], [-1., 2.], [0., torch.nan]])
        figure = plot_heatmaps(probe, senders, effects, ["one", "two"])
        mean_grid = figure.axes[2].images[0].get_array()
        self.assertAlmostEqual(mean_grid[0, 0], 1.25)
        self.assertAlmostEqual(mean_grid[0, 1], .5)
        self.assertAlmostEqual(mean_grid[1, 0], 0.)
        self.assertTrue(mean_grid.mask[1, 1])
        self.assertEqual(figure.axes[0].images[0].norm.vmax, 2.)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
