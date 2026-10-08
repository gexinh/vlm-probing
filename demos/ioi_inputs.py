"""Author IOI tokenization and candidate-margin contract."""
from dataclasses import dataclass
import torch
@dataclass
class BatchCandidateMargin:
    """Different IO/subject token IDs per example, at the last prompt token."""
    positive: list[int]
    negative: list[int]

    def score(self, logits, layout):
        chosen = layout.select("last_prompt")
        if not (chosen.sum(-1) == 1).all() or len(self.positive) != logits.shape[0]:
            raise ValueError("one prediction position and one candidate pair per example are required")
        values = logits[chosen].float()
        rows = torch.arange(values.shape[0], device=values.device)
        return values[rows, torch.tensor(self.positive, device=values.device)] - values[
            rows, torch.tensor(self.negative, device=values.device)]

def prepare_pairs(pairs, tokenizer, device):
    """Native GPT-2 text pairs, with the author TransformerLens BOS convention."""
    from vlm_probing import ProbeInputs, TokenLayout
    versions = []
    for key in ("clean_prompt", "donor_prompt"):
        sequences = [[tokenizer.bos_token_id, *tokenizer(pair[key], add_special_tokens=False)["input_ids"]]
                     for pair in pairs]
        width = max(map(len, sequences))
        ids = torch.full((len(pairs), width), tokenizer.eos_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for row, sequence in enumerate(sequences):
            ids[row, :len(sequence)] = torch.tensor(sequence, device=device)
            mask[row, :len(sequence)] = True
        layout = TokenLayout(mask, torch.zeros_like(mask), mask, ids)
        versions.append(ProbeInputs({"input_ids": ids, "attention_mask": mask.long()}, layout))
    if not torch.equal(versions[0].layout.valid, versions[1].layout.valid):
        raise ValueError("clean/donor token positions must align")
    for pair in pairs:
        for prefix, text in (("positive", pair["positive_token"]), ("negative", pair["negative_token"])):
            if tokenizer(text, add_special_tokens=False)["input_ids"] != [pair[f"{prefix}_id"]]:
                raise ValueError("author candidate token IDs do not match this tokenizer")
    return (*versions, BatchCandidateMargin([p["positive_id"] for p in pairs], [p["negative_id"] for p in pairs]))
