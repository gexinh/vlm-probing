"""Run all 18 public factories on a real tiny decoder, without downloads.

    PYTHONPATH=src python examples/prober_quickstart.py
"""
import tempfile

import torch

from tiny_model import TinyModel
from vlm_probing import Prober, TokenMargin


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = TinyModel().eval()
    probe = Prober(model)
    clean = {"input_ids": torch.tensor([[1, 2, 3], [3, 4, 5]]),
             "image_tokens": torch.randn(2, 2, 6)}
    corrupt = {**clean, "image_tokens": torch.zeros_like(clean["image_tokens"])}
    metric = TokenMargin(positive=4, negative=5)
    print("Available:", sum(v["available"] for v in probe.describe()["methods"].values()))

    result = probe.lens.logit(layers=[0, 1]).run(clean)
    print("Logit Lens [layer, selected position, vocabulary]:", result.tensors["logits"].shape)
    print("Positions [batch, expanded token]:", result.tensors["positions"].tolist())
    probe.lens.embed().run(clean, top_k=3)
    binding = {"model_id": "tiny-seed7", "tokenizer_id": "integer-demo",
               "calibration_id": "two-demo-prompts", "readout_id": "tiny.norm+head"}
    tuned = probe.lens.tuned(layers=[0], binding=binding).fit(clean, steps=20, lr=0.01)
    print("Tuned calibration KL:", tuned.losses[0][0], "->", tuned.losses[0][-1])
    with tempfile.TemporaryDirectory() as path:
        tuned.save(path)
        probe.lens.tuned(layers=[0], binding=binding).load(path).run(clean)
    probe.lens.attention(layers=[0]).fit(clean, steps=5).run(clean)
    probe.lens.jacobian(layers=[0]).fit(clean, skip_first=0, exclude_last=False).run(clean)
    probe.lens.patchscope(layers=[0], source_position=0, target_layer=0, target_position=2).run(
        clean, target_inputs={**clean, "input_ids": torch.tensor([[6, 7, 8], [6, 7, 8]])})

    patched = probe.causal.patch(layers=[0, 1], tokens="visual").run(corrupt, source=clean, metric=metric)
    print("Patch effects [independent layer, batch]:", patched.tensors["effect"])
    probe.causal.ablate(layers=[0]).run(clean, metric=metric)
    probe.causal.attribute(layers=[0]).run(corrupt, source=clean, metric=metric)
    probe.causal.knockout(layers=[0]).run(clean, metric=metric)
    probe.causal.steer(layers=[0], strength=0.1).run(clean, direction=torch.ones(6), metric=metric)
    probe.causal.eap_ig(steps=4).run(corrupt, source=clean, metric=metric)

    probe.attention.profile(layers=[0, 1]).run(clean)
    probe.attention.rollout().run(clean)
    probe.attention.relevance().run(clean, metric=metric)
    probe.attention.head_logits(layers=[1]).run(clean)
    probe.attention.reweight(layers=[0], weight=2.).run(clean, metric=metric)
    probe.attention.temperature(layers=[0], temperature=0.7).run(clean, metric=metric)
    print("All 18 model-bound methods completed.")


if __name__ == "__main__":
    main()
