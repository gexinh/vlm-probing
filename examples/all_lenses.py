"""Six lens execution paths on a real tiny decoder; not paper benchmarks.

Run: PYTHONPATH=src python examples/all_lenses.py
"""
import torch

from vlm_probing.adapters import HookPoint, ModelReadout, TorchModelAdapter
from vlm_probing.lenses import (
    AttentionLens, EmbedLens, JacobianLens, LogitLens, Patchscope, TunedLens,
)
from tiny_model import TinyModel


def main():
    torch.manual_seed(11)
    torch.set_num_threads(1)
    model = TinyModel().eval()
    readout = ModelReadout(model.lm_head, norm=model.norm)
    adapter = TorchModelAdapter(model, {
        "layer0": HookPoint("layers.0"), "layer1": HookPoint("layers.1"),
        "head0": HookPoint("layers.0.attention.output"),
    })
    batch = {"input_ids": torch.tensor([[1, 2, 3], [3, 4, 5]]),
             "image_tokens": torch.randn(2, 2, 6)}
    trace = adapter.run(batch, capture=["layer0", "head0"])
    hidden, teacher = trace.activations["layer0"], trace.logits
    binding = {"model_id": "tiny-seed11", "site": "layer0.residual",
               "readout_id": "tiny.norm+lm_head", "tokenizer_id": "integer-fixture",
               "calibration_id": "two-local-prompts"}

    print("LogitLens:", tuple(LogitLens(readout).run(hidden).tensors["logits"].shape))
    print("EmbedLens:", EmbedLens(model.embedding.weight).run(batch["image_tokens"], top_k=2)
          .tensors["token_ids"].tolist())

    tuned = TunedLens(readout, 6, binding=binding)
    losses = tuned.fit(hidden, teacher, steps=20, lr=0.01)
    print("TunedLens calibration KL:", round(losses[0], 6), "->", round(losses[-1], 6))
    tuned.run(hidden)

    # The fixture has one query head, already output-projected into model width.
    head_outputs = trace.activations["head0"].unsqueeze(-2)  # [B,S,H,D]
    head_lens = AttentionLens(1, 6, 12, initial_unembedding=model.lm_head.weight,
                             binding={**binding, "site": "layer0.projected_head"})
    head_lens.fit(head_outputs, teacher, steps=20, lr=0.01)
    print("AttentionLens:", tuple(head_lens.run(head_outputs).tensors["head_logits"].shape))

    jacobian = JacobianLens(readout, binding=binding)
    jacobian.fit(hidden, model.layers[1], skip_first=0, exclude_last=False)
    print("JacobianLens:", tuple(jacobian.run(hidden).tensors["logits"].shape))

    def target_runner(patch, *, target_layer, target_position, target_inputs):
        def replace(hidden):
            if patch.shape != hidden[:, target_position].shape:
                raise ValueError("Source/target patch shapes do not match")
            edited = hidden.clone()
            edited[:, target_position] = patch
            return edited
        return adapter.run(target_inputs, interventions={f"layer{target_layer}": replace}).logits

    scope = Patchscope(target_runner, source_model_id="tiny", target_model_id="tiny")
    target = {**batch, "input_ids": torch.tensor([[6, 7, 8], [6, 7, 8]])}
    result = scope.run(hidden, source_position=2, target_position=2,
                       target_layer=0, target_inputs=target, layer=0)
    print("Patchscope target-prompt logits:", tuple(result.tensors["logits"].shape))


if __name__ == "__main__":
    main()
