"""Input-embedding nearest neighbours: https://github.com/EIT-NLP/EmbedLens."""
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .base import BaseLens, floating_tensor
from ..core.types import ProbeResult


class EmbedLens(BaseLens):
    """Cosine nearest neighbours of projected visual tokens in input embeddings.

    This implements the official notebook's semantic readout, not the paper's
    full clustering/pruning evaluation. Optional token groups are explicit
    checkpoint-specific rules on the nearest token; no universal sink/dead IDs
    are assumed. Unmatched tokens get label -1, not an asserted 'alive' label.
    """

    name = "embed_lens"

    def __init__(
        self, input_embeddings: Tensor, *,
        token_groups: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        floating_tensor(input_embeddings, "input_embeddings", ndim=2)
        self.embeddings = input_embeddings.detach()
        self.token_groups = {name: tuple(ids) for name, ids in (token_groups or {}).items()}
        seen: set[int] = set()
        for name, ids in self.token_groups.items():
            if not name or not ids or any(not isinstance(i, int) or i < 0 or i >= len(input_embeddings) for i in ids):
                raise ValueError("token groups require nonempty names and valid vocabulary IDs")
            if seen.intersection(ids):
                raise ValueError("token groups must be disjoint")
            seen.update(ids)

    @torch.no_grad()
    def run(
        self, activations: Tensor, *, top_k: int = 10,
        layer: int | None = None, vocabulary_chunk_size: int = 8192,
    ) -> ProbeResult:
        floating_tensor(activations, "activations")
        if activations.shape[-1] != self.embeddings.shape[-1]:
            raise ValueError("activations and input embeddings must share their hidden width")
        if not isinstance(top_k, int) or not 1 <= top_k <= len(self.embeddings):
            raise ValueError("top_k must be between 1 and vocabulary size")
        if not isinstance(vocabulary_chunk_size, int) or vocabulary_chunk_size < 1:
            raise ValueError("vocabulary_chunk_size must be positive")
        query = F.normalize(activations.float(), dim=-1)
        values = query.new_empty((*query.shape[:-1], 0))
        indices = torch.empty_like(values, dtype=torch.long)
        # Keep only top k candidates from each vocabulary block.
        for start in range(0, len(self.embeddings), vocabulary_chunk_size):
            embeddings = self.embeddings[start:start + vocabulary_chunk_size].to(query.device).float()
            scores = query @ F.normalize(embeddings, dim=-1).T
            candidates, ids = scores.topk(min(top_k, scores.shape[-1]), dim=-1)
            all_values = torch.cat((values, candidates), dim=-1)
            all_indices = torch.cat((indices, ids + start), dim=-1)
            values, locations = all_values.topk(min(top_k, all_values.shape[-1]), dim=-1)
            indices = all_indices.gather(-1, locations)
        tensors = {"token_ids": indices, "similarities": values,
                   "activation_norms": activations.float().norm(dim=-1)}
        if self.token_groups:
            labels = torch.full_like(indices[..., 0], -1)
            for label, ids in enumerate(self.token_groups.values()):
                members = torch.tensor(ids, device=indices.device)
                labels[torch.isin(indices[..., 0], members)] = label
            tensors["group_labels"] = labels
        return ProbeResult(self.name, tensors, {
            "layer": layer, "metric": "cosine", "space": "input_embeddings",
            "group_names": list(self.token_groups), "unmatched_group_label": -1,
            "scope": "nearest_token_semantic_readout",
        })
