"""Dataset and plotting helpers for the portable path-patching notebook.

Only pairing and model input preparation are dataset-dependent. The intervention
itself remains ``probe.causal.path(...).sweep(...)`` for every supported adapter.
"""
import json
import math
from pathlib import Path

import torch

from vlm_probing import CapabilityError, TokenMargin


class _Float32TokenMargin(TokenMargin):
    def score(self, logits, layout):
        if not 0 <= self.positive < logits.shape[-1] or not 0 <= self.negative < logits.shape[-1]:
            raise ValueError("Candidate token ID is outside the vocabulary")
        candidates = logits[..., [self.positive, self.negative]].float()
        return TokenMargin(0, 1, position=self.position).score(candidates, layout)


class _PreparedInputs(dict):
    """Native kwargs with an encoder attribute, never an extra model argument."""

    def __init__(self, kwargs, encode_continuation):
        super().__init__(kwargs)
        self.encode_continuation = encode_continuation


def load_pairs(path):
    """Read canonical JSONL pairs; resolve relative image paths beside the file.

    Each record needs a unique ``id`` and the four nonempty string fields
    ``clean_prompt``, ``donor_prompt``, ``clean_answer``, and ``donor_answer``.
    Optional ``clean_image`` / ``donor_image`` and ``source`` preserve provenance.
    Construct counterfactual prompts and answers in the dataset's own adapter;
    there is no universal corruption rule shared by all tasks.
    """
    path, pairs, seen = Path(path), [], set()
    fields = ("id", "clean_prompt", "donor_prompt", "clean_answer", "donor_answer")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number}: each pair must be an object")
        if any(not isinstance(record.get(key), str) or not record[key].strip() for key in fields):
            raise ValueError(f"line {line_number}: required nonempty strings: {', '.join(fields)}")
        if record["id"] in seen:
            raise ValueError(f"duplicate pair id: {record['id']}")
        if record["clean_answer"] == record["donor_answer"]:
            raise ValueError(f"pair {record['id']}: clean and donor answers must differ")
        if "source" in record and not isinstance(record["source"], dict):
            raise ValueError(f"pair {record['id']}: source must be an object")
        for key in ("clean_image", "donor_image"):
            if record.get(key) is not None:
                if not isinstance(record[key], str) or not record[key].strip():
                    raise ValueError(f"pair {record['id']}: {key} must be an image path")
                image_path = Path(record[key])
                record[key] = str(image_path if image_path.is_absolute() else path.parent / image_path)
        seen.add(record["id"])
        pairs.append(record)
    if not pairs:
        raise ValueError("the pair file is empty")
    return pairs


def load_model(model_id, device="cuda:0", dtype=None, attn_implementation="sdpa"):
    """Load a native Hugging Face LLM or VLM and its processor.

    Loading and dataset conversion are separate from the probing API. A model
    still needs an adapter with complete editable head outputs for path patching;
    call ``head_grid(probe)`` before starting a sweep to check that capability.
    Path patching edits head messages and does not need attention probabilities,
    so SDPA is the default. Pass ``attn_implementation='eager'`` when desired.
    """
    from transformers import (AutoConfig, AutoModelForCausalLM,
                              AutoModelForImageTextToText, AutoProcessor, AutoTokenizer)

    config = AutoConfig.from_pretrained(model_id)
    is_vlm = getattr(config, "vision_config", None) is not None
    model_class = AutoModelForImageTextToText if is_vlm else AutoModelForCausalLM
    processor_class = AutoProcessor if is_vlm else AutoTokenizer
    dtype = dtype or (torch.bfloat16 if str(device).startswith("cuda") else torch.float32)
    model = model_class.from_pretrained(model_id, dtype=dtype, attn_implementation=attn_implementation,
                                        device_map=device).eval()
    return model, processor_class.from_pretrained(model_id)


def prepare_inputs(pair, processor, device):
    """Apply the native chat template and return clean/donor model kwargs.

    Text-only VLM tasks render structured text with the processor's native chat
    template, then tokenize without a fabricated image. With an image, use the
    multimodal processor and its image placeholder. A missing
    donor image reuses the clean image, so text counterfactuals preserve vision.
    The pair must still satisfy the library's explicit sequence alignment rule.
    Base LLM checkpoints without a chat template receive the original prompt
    unchanged; no system message or invented conversation format is inserted.
    """
    from PIL import Image

    tokenizer = getattr(processor, "tokenizer", processor)
    is_multimodal = hasattr(processor, "tokenizer")
    processor_template = getattr(processor, "chat_template", None) if is_multimodal else None
    template = processor_template or getattr(tokenizer, "chat_template", None)
    # Let the native processor select its own default when it has named templates.
    template_kwargs = ({} if processor_template else {"chat_template":
                       tokenizer.get_chat_template() if hasattr(tokenizer, "get_chat_template") and template
                       else template})
    system = pair.get("source", {}).get("system_prompt", "You are a helpful assistant")
    if not isinstance(system, str) or not system.strip():
        raise ValueError("system_prompt must be a nonempty string")

    def prepare(prompt, image):
        if image is None:
            if is_multimodal and template:
                messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                            {"role": "user", "content": [{"type": "text", "text": prompt}]}]
                text = processor.apply_chat_template(messages, **template_kwargs,
                                                     tokenize=False, add_generation_prompt=True)
            elif template:
                messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = prompt
            if prompt not in text:
                raise ValueError("the native chat template omitted the user prompt")
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)

            def encode_continuation(candidate):
                return tokenizer(text + candidate, return_tensors="pt",
                                 add_special_tokens=False)["input_ids"]
        else:
            if not hasattr(processor, "tokenizer"):
                raise ValueError("image pairs need a multimodal processor")
            if isinstance(image, (str, Path)):
                with Image.open(image) as opened:
                    image = opened.convert("RGB")
            if not isinstance(image, Image.Image):
                raise TypeError("images must be a path or PIL image")
            messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                        {"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(messages, **template_kwargs,
                                                 tokenize=False, add_generation_prompt=True)
            if prompt not in text:
                raise ValueError("the native chat template omitted the user prompt")
            inputs = processor(text=[text], images=[image], return_tensors="pt")

            def encode_continuation(candidate):
                return processor(text=[text + candidate], images=[image],
                                 return_tensors="pt")["input_ids"]
        return _PreparedInputs(
            {key: value.to(device) if isinstance(value, torch.Tensor) else value
             for key, value in inputs.items()}, encode_continuation)

    clean_image = pair.get("clean_image")
    donor_image = pair.get("donor_image", clean_image)
    if donor_image is None:
        donor_image = clean_image
    return (prepare(pair["clean_prompt"], clean_image),
            prepare(pair["donor_prompt"], donor_image))


def head_grid(probe):
    """Enumerate actual query heads, rejecting incomplete path capabilities."""
    capability = probe.describe()["methods"]["causal.path"]
    if not capability["available"]:
        raise CapabilityError(f"path patching is unavailable: {capability['missing']}")
    return [(layer, head) for layer, point in sorted(probe.spec.path_heads.items())
            for head in range(point.heads)]


def first_answer_metric(pair, tokenizer, answer_prefix="", *, clean_inputs=None, donor_inputs=None):
    """Return a float32 first-answer-token margin and the exact candidate IDs.

    This is a first-token approximation, not a sequence likelihood. An answer
    prefix (for example a space) is an explicit experimental choice. Pass both
    prepared input dictionaries to derive continuation tokens in their actual
    contexts with the same native preprocessing and unchanged prompt boundaries.
    Plain input dictionaries instead need an exact decode/encode round trip. Both
    contexts must use the same first token for each candidate. Without inputs,
    this explicitly uses standalone ``prefix + answer`` tokenization, which can
    differ from actual continuations for SentencePiece and other tokenizers.
    """
    candidates = [answer_prefix + pair[key] for key in ("clean_answer", "donor_answer")]
    if (clean_inputs is None) != (donor_inputs is None):
        raise ValueError("provide both clean_inputs and donor_inputs for context-checked scoring")
    context_checked = clean_inputs is not None
    context_encoders = []
    if context_checked:
        context_ids = []
        for inputs in (clean_inputs, donor_inputs):
            input_ids = inputs.get("input_ids")
            if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
                raise ValueError("context-checked first-token scoring requires input_ids with batch size 1")
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                if attention_mask.shape != input_ids.shape:
                    raise ValueError("attention_mask must match input_ids")
                input_ids = input_ids[attention_mask.bool()].reshape(1, -1)
            prefix_ids = input_ids[0].detach().cpu().tolist()
            if not prefix_ids:
                raise ValueError("context-checked scoring requires a nonempty prompt")
            encoder = getattr(inputs, "encode_continuation", None)
            if encoder is None:
                prompt = tokenizer.decode(prefix_ids, skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False)
                if tokenizer.encode(prompt, add_special_tokens=False) != prefix_ids:
                    raise ValueError("prompt tokens do not round-trip; use prepare_inputs or provide "
                                     "a custom continuation metric")
                context_encoders.append("decode_roundtrip")
            else:
                context_encoders.append("native_input_encoder")
            continuations = []
            for candidate in candidates:
                if encoder is None:
                    complete_ids = tokenizer.encode(prompt + candidate, add_special_tokens=False)
                else:
                    complete_ids = encoder(candidate)
                    if (not isinstance(complete_ids, torch.Tensor) or complete_ids.ndim != 2
                            or complete_ids.shape[0] != 1):
                        raise ValueError("native continuation encoder must return input_ids with batch size 1")
                    complete_ids = complete_ids[0].detach().cpu().tolist()
                if complete_ids[:len(prefix_ids)] != prefix_ids:
                    raise ValueError("answer changes the prompt token boundary; choose an explicit answer "
                                     "prefix or provide a custom continuation metric")
                continuations.append(complete_ids[len(prefix_ids):])
            if not all(continuations):
                raise ValueError("answer candidates must contain at least one continuation token")
            context_ids.append(continuations)
        if any(clean[0] != donor[0] for clean, donor in zip(*context_ids)):
            raise ValueError("candidate first-token IDs differ across clean/donor contexts; "
                             "provide a custom continuation metric")
        ids = context_ids[0]
    else:
        ids = [tokenizer.encode(candidate, add_special_tokens=False) for candidate in candidates]
    if not all(ids):
        raise ValueError("answer candidates must contain at least one token")
    if ids[0][0] == ids[1][0]:
        raise ValueError("answer candidates share their first token; use a sequence metric instead")
    margin = _Float32TokenMargin(ids[0][0], ids[1][0], position="last_prompt")
    return margin, {
        "clean_candidate": candidates[0], "donor_candidate": candidates[1],
        "clean_token_ids": ids[0], "donor_token_ids": ids[1],
        "positive": ids[0][0], "negative": ids[1][0],
        "answer_prefix": answer_prefix, "prediction_position": "last_prompt",
        "scoring": "float32 first-answer-token logit margin",
        "candidate_tokenization": "context_checked" if context_checked else "standalone",
        "context_encoders": context_encoders,
    }


def normalized_effect(result, epsilon=1e-6):
    """Return ratios [sender, batch] and eligible [batch] without clipping.

    Compute ``(patched - base) / (donor - base)`` separately for each example.
    A denominator smaller than ``epsilon`` is undefined for this normalized
    statistic; preserve its column as NaN instead of manufacturing zero effect.
    """
    if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    base = result.tensors["baseline_score"].detach().float().cpu().reshape(-1)
    donor = result.tensors["donor_score"].detach().float().cpu().reshape(-1)
    patched = result.tensors["intervention_score"].detach().float().cpu()
    if patched.ndim == 1 and base.numel() == 1:
        patched = patched[:, None]
    if patched.ndim != 2 or patched.shape[1] != base.numel() or donor.shape != base.shape:
        raise ValueError("expected shared baseline/donor [batch] and interventions [sender,batch]")
    if not all(torch.isfinite(value).all() for value in (base, donor, patched)):
        raise ValueError("raw path scores must be finite")
    denominator = donor - base
    eligible = denominator.abs() >= epsilon
    values = torch.full_like(patched, torch.nan)
    values[:, eligible] = (patched[:, eligible] - base[eligible]) / denominator[eligible]
    if not torch.isfinite(values[:, eligible]).all():
        raise ValueError("normalized effects overflowed; inspect the raw scores")
    return values, eligible


def effect_grid(probe, senders, values):
    """Map senders to a CPU grid using a Prober or saved layer-to-head counts."""
    from collections.abc import Mapping

    senders = [tuple(sender) for sender in senders]
    values = torch.as_tensor(values).detach().float().cpu()
    if values.ndim != 1 or len(senders) != values.numel() or len(set(senders)) != len(senders):
        raise ValueError("one value is required for each unique sender coordinate")
    points = probe if isinstance(probe, Mapping) else probe.spec.path_heads
    counts = {layer: point if type(point) is int else point.heads
              for layer, point in points.items()}
    if not counts:
        raise CapabilityError("adapter has no path head outputs")
    if any(type(layer) is not int or layer < 0 or type(count) is not int or count <= 0
           for layer, count in counts.items()):
        raise ValueError("head counts need nonnegative integer layers and positive integer counts")
    grid = torch.full((max(counts) + 1, max(counts.values())), torch.nan)
    for sender, value in zip(senders, values):
        if (len(sender) != 2 or any(type(coordinate) is not int for coordinate in sender)
                or sender[0] not in counts or not 0 <= sender[1] < counts[sender[0]]):
            raise ValueError(f"sender coordinate is unavailable: {sender}")
        layer, head = sender
        grid[layer, head] = value
    return grid


def plot_heatmaps(probe, senders, effects, labels=None, *, pairs=None, model_name=None,
                  highlight_count=3):
    """Compatibility display using the public causal-heatmap renderer.

    Real coordinates, per-pair signs and undefined measurements are preserved.
    Canonical prompt cards are displayed separately by the notebook.
    """
    import matplotlib.pyplot as plt
    from vlm_probing.visualization import plot_causal_heatmap
    from demos.path_display import sample_title
    if pairs is not None:
        labels = [sample_title(pair,index).split("\n")[0] for index,pair in enumerate(pairs)]
    if labels is None:
        raise ValueError("provide labels or canonical pair records")
    effects = torch.as_tensor(effects).detach().float().cpu()
    if effects.ndim != 2 or effects.shape != (len(senders),len(labels)):
        raise ValueError("effects must be [sender,pair], matching senders and labels")
    if torch.isinf(effects).any() or not torch.isfinite(effects).any():
        raise ValueError("heatmaps require at least one finite effect and no infinite values")
    grids = [effect_grid(probe,senders,effects[:,index]) for index in range(len(labels))]
    grids.append(effect_grid(probe,senders,effects.nanmean(dim=1)))
    limit = max(effects[torch.isfinite(effects)].abs().max().item(),1e-6)
    fig,axes=plt.subplots(1,len(grids),figsize=(5*len(grids),5),squeeze=False,constrained_layout=True)
    for ax,grid,label in zip(axes[0],grids,[*labels,"Mean across displayed pairs"]):
        plot_causal_heatmap(grid,ax=ax,title=label,vmin=-limit,vmax=limit,colorbar=False,
                            value_label="Normalized path effect",highlight_count=highlight_count)
    fig.colorbar(axes[0,-1].images[0],ax=axes.ravel().tolist(),label="Normalized path effect")
    return fig
