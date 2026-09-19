"""Run: PYTHONPATH=src python examples/quickstart.py (CPU; no downloads)."""
import torch

from vlm_probing.adapters import HookPoint, ModelReadout, TorchModelAdapter
from vlm_probing.attention import AttentionProfile
from vlm_probing.causal import ActivationPatching, AttentionKnockout
from vlm_probing.lenses import EmbedLens, LogitLens
from vlm_probing.metrics import token_logit_margin
from tiny_model import TinyModel


def main():
    torch.manual_seed(7)
    model = TinyModel().eval()
    readout = ModelReadout(model.lm_head, norm=model.norm)
    adapter = TorchModelAdapter(model, {
        "layer0": HookPoint("layers.0"),
        "layer1": HookPoint("layers.1"),
        "attention0": HookPoint("layers.0.attention.probs"),
        "scores0": HookPoint("layers.0.attention.scores"),
    }, model_id="local/tiny-fixture", readout=readout)
    clean = {"input_ids": torch.tensor([[1, 2, 3]]), "image_tokens": torch.randn(1, 2, 6)}
    corrupt = {**clean, "image_tokens": torch.zeros_like(clean["image_tokens"])}
    trace = adapter.run(clean, capture=["layer0", "layer1", "attention0"])

    lens = LogitLens(readout)
    layer_logits = lens.run(trace.activations["layer1"], layer=1).tensors["logits"]
    torch.testing.assert_close(layer_logits, trace.logits)
    print("Final-layer logit lens matches model logits.")

    embedding_lens = EmbedLens(model.embedding.weight)
    neighbors = embedding_lens.run(clean["image_tokens"], top_k=3)
    print("Visual-token embedding neighbors:", neighbors.tensors["token_ids"].tolist())

    visual = torch.tensor([[True, True, False, False, False]])
    profile = AttentionProfile().run(trace.activations["attention0"], groups={"visual": visual})
    print("Last-query visual attention mass:", profile.tensors["group_mass"][0, 0, -1].tolist())

    patcher = ActivationPatching()
    patched = adapter.run(corrupt, interventions={
        "layer0": lambda receiver: patcher.run(receiver, trace.activations["layer0"],
                                               mask=visual[..., None]).tensors["edited"]
    })
    baseline = adapter.run(corrupt)
    margin = lambda output: token_logit_margin(output.logits[:, -1], 4, 5)
    print("Visual-state patch margin change:", (margin(patched) - margin(baseline)).tolist())

    blocked = torch.zeros(1, 1, 5, 5, dtype=torch.bool)
    blocked[..., -1, :2] = True
    knockout = AttentionKnockout()
    intervention = adapter.run(clean, capture=["attention0"], interventions={
        "scores0": lambda logits: knockout.run(logits, blocked=blocked).tensors["edited"]
    })
    assert intervention.activations["attention0"][..., -1, :2].eq(0).all()
    print("Attention knockout blocked real query-to-image edges before value aggregation.")
    print("Knockout margin change:", (margin(intervention) - margin(trace)).tolist())


if __name__ == "__main__":
    main()
