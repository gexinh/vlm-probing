"""A small real attention computation for CPU examples, not a pretrained VLM.

Image-token vectors are supplied explicitly; this fixture has no image encoder.
"""
import math
import torch
from torch import Tensor, nn


class TinyAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.scores = nn.Identity()
        self.probs = nn.Softmax(dim=-1)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = (q @ k.transpose(-2, -1))[:, None] / math.sqrt(x.shape[-1])
        future = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device).triu(1)
        scores = self.scores(scores.masked_fill(future, -torch.inf))
        probs = self.probs(scores)
        return self.output((probs @ v[:, None]).squeeze(1))


class TinyBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.attention = TinyAttention(width)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(x)
        return x + self.mlp(x)


class TinyModel(nn.Module):
    def __init__(self, width: int = 6, vocab_size: int = 12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.layers = nn.ModuleList([TinyBlock(width), TinyBlock(width)])
        self.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)

    def forward(self, input_ids: Tensor, image_tokens: Tensor) -> dict[str, Tensor]:
        hidden = torch.cat([image_tokens, self.embedding(input_ids)], dim=1)
        for layer in self.layers:
            hidden = layer(hidden)
        return {"logits": self.lm_head(self.norm(hidden))}

    def probing_adapter(self):
        """An explicit model protocol: Prober(model) uses this verified contract."""
        from vlm_probing import HookPoint, ModelReadout, ModelSpec, TokenLayout, TorchModelAdapter

        sites, residuals, attentions, scores, heads = {}, {}, {}, {}, {}
        for i in range(len(self.layers)):
            residuals[i], attentions[i] = f"residual.{i}", f"attention.{i}"
            scores[i], heads[i] = f"scores.{i}", f"heads.{i}"
            sites[residuals[i]] = HookPoint(f"layers.{i}")
            sites[attentions[i]] = HookPoint(f"layers.{i}.attention.probs")
            sites[scores[i]] = HookPoint(f"layers.{i}.attention.scores")
            sites[heads[i]] = HookPoint(f"layers.{i}.attention.output")
        sites["embeddings"] = HookPoint("layers.0", kind="input", selector=0)
        sites["last_mlp"] = HookPoint(f"layers.{len(self.layers)-1}.mlp")

        def layout(inputs):
            ids = inputs["input_ids"]
            prefix = torch.zeros(inputs["image_tokens"].shape[:2], dtype=ids.dtype, device=ids.device)
            expanded = torch.cat([prefix, ids], dim=1)
            valid = torch.ones_like(expanded, dtype=torch.bool)
            visual = torch.zeros_like(valid)
            visual[:, :prefix.shape[1]] = True
            return TokenLayout(valid, visual, token_ids=expanded)

        def linear_readout(full):
            scale = (full.var(-1, keepdim=True, unbiased=False) + self.norm.eps).rsqrt()
            return self.lm_head.weight * self.norm.weight, scale, True

        spec = ModelSpec(
            residuals, attentions=attentions, attention_scores=scores, heads=heads,
            embeddings="embeddings", input_embeddings=self.embedding.weight,
            layout=layout, project_heads=lambda _, value: value.unsqueeze(-2),
            linear_readout=linear_readout, editable_attention=set(attentions),
            edges={"last_attention_to_residual": heads[len(self.layers)-1],
                   "last_mlp_to_residual": "last_mlp"},
            notes=("Random CPU demonstration model; image embeddings are supplied directly.",),
        )
        return TorchModelAdapter(self, sites, readout=ModelReadout(self.lm_head, norm=self.norm),
                                 model_id="tiny-demo", spec=spec)
