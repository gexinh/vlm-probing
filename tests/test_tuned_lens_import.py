"""Author lens weights cannot be bound to altered GPT-2 computation."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

try:
    from transformers import GPT2Config, GPT2LMHeadModel
    from safetensors.torch import save_file
except ImportError:
    GPT2Config = None

from vlm_probing import Prober
from vlm_probing.training import import_author_tuned_lens


@unittest.skipIf(GPT2Config is None, "requires the Transformers extra")
class TunedLensImportTests(unittest.TestCase):
    def test_changed_head_partition_is_rejected_despite_identical_parameters(self):
        config = GPT2Config(n_embd=12, n_layer=2, n_head=3, n_positions=16, vocab_size=17)
        original = GPT2LMHeadModel(config)
        altered = GPT2LMHeadModel(GPT2Config(**{**config.to_dict(), "n_head": 6}))
        altered.load_state_dict(original.state_dict())
        for name, parameter in original.named_parameters():
            torch.testing.assert_close(parameter, dict(altered.named_parameters())[name])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            base.mkdir()
            config.to_json_file(base / "config.json")
            weights = base / "model.safetensors"
            save_file({name: parameter.detach().contiguous() for name, parameter in original.transformer.named_parameters()}, weights)
            (root / "config.json").write_text(json.dumps({"lens_type": "linear_tuned_lens",
                "d_model": 12, "num_hidden_layers": 2, "base_model_name_or_path": "toy-gpt2",
                "base_model_revision": "toy@1"}))
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "configuration differs at n_head"):
                import_author_tuned_lens(Prober(altered), root, base_weight_file=weights,
                    expected_weight_sha256=digest, tokenizer_id="toy-tokenizer@1")


if __name__ == "__main__":
    unittest.main()
