"""Measure selected sender-to-receiver paths, separately from head-to-output scans.

The reported edges are controlled interventions through a receiver's query
activation. They are not direct structural connections or a recovered circuit.
Candidate selection uses the existing head-to-output ranking on the same examples;
the resulting path measurements are exploratory and have no held-out validation.
"""
import math

from demos.path_helpers import first_answer_metric, normalized_effect, prepare_inputs


def select_candidates(ranking, head_counts, sender_count=4, receiver_count=2):
    """Select late receivers and strictly earlier senders without model constants.

    ``ranking`` contains layer/head/mean_effect records. ``head_counts`` maps
    decoder layer indices to actual query-head counts. Receiver candidates are
    the last ceil(10% of decoder layers), ranked by absolute mean head-to-output
    effect. Senders use the same ranking but precede every selected receiver.
    Ties are broken by layer and head to keep configuration reproducible.
    """
    if (not head_counts or any(type(layer) is not int or layer < 0 or
                              type(count) is not int or count < 1
                              for layer, count in head_counts.items())):
        raise ValueError("head_counts must map nonnegative layers to positive counts")
    if any(type(count) is not int or count < 1 for count in (sender_count, receiver_count)):
        raise ValueError("sender_count and receiver_count must be positive integers")
    seen, candidates = set(), []
    for item in ranking:
        layer, head = item["layer"], item["head"]
        if (type(layer) is not int or type(head) is not int or layer not in head_counts
                or not 0 <= head < head_counts[layer]):
            raise ValueError(f"ranked head is unavailable: {(layer, head)}")
        coordinate = (layer, head)
        if coordinate in seen:
            raise ValueError(f"duplicate ranked head: {coordinate}")
        seen.add(coordinate)
        value = item["mean_effect"]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("ranked effects must be finite numbers")
        candidates.append((coordinate, float(value)))
    ordered = sorted(candidates, key=lambda item: (-abs(item[1]), *item[0]))
    layers = sorted(head_counts)
    late_layers = set(layers[-max(1, math.ceil(len(layers) * 0.1)):])
    receivers = [coordinate for coordinate, _ in ordered if coordinate[0] in late_layers][:receiver_count]
    if len(receivers) != receiver_count:
        raise ValueError("not enough ranked heads in the last 10% of decoder layers")
    first_receiver = min(layer for layer, _ in receivers)
    senders = [coordinate for coordinate, _ in ordered if coordinate[0] < first_receiver][:sender_count]
    if len(senders) != sender_count:
        raise ValueError("not enough ranked sender heads before every selected receiver")
    return senders, [(layer, head, "q") for layer, head in receivers]


def aggregate_pair_effects(per_pair):
    """Average eligible per-pair ratios, preserving valid zero effects.

    An ineligible denominator is represented by ``normalized_effect=None``.
    Raw scores are always retained. This never divides averaged raw changes by
    an averaged denominator, and it never drops zero-valued measurements.
    """
    values = [item["normalized_effect"] for item in per_pair
              if item["normalized_effect"] is not None]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("normalized effects must be finite or None")
    return sum(values) / len(values) if values else None


def measure_receiver_paths(probe, pairs, processor, device, senders, receivers, *, epsilon=1e-6):
    """Measure each candidate sender through one receiver Q, then to the output.

    The controlled pass freezes all attention-head outputs except the selected
    sender. MLP/LayerNorm recompute. A fresh pass injects only the recomputed
    receiver Q. Scoring and intervention positions are the last prompt token.
    Native chat preprocessing and answer-context checks are shared with the main
    demo. Identity-donor and reverse-layer controls run for every example.

    Returns a JSON-ready artifact. All Q endpoints are validated before the first
    model forward by the existing ``causal.path`` factory. An unsupported adapter
    fails explicitly rather than silently switching intervention semantics.
    """
    if not pairs:
        raise ValueError("at least one clean/donor pair is required")
    senders, receivers = [tuple(item) for item in senders], [tuple(item) for item in receivers]
    if not senders or not receivers:
        raise ValueError("nonempty sender and receiver selections are required")
    if any(len(receiver) != 3 or receiver[2] != "q" for receiver in receivers):
        raise ValueError("this demonstration measures receiver Q endpoints")
    if len(set(senders)) != len(senders) or len(set(receivers)) != len(receivers):
        raise ValueError("sender and receiver selections must be unique")
    if any(sender[0] >= receiver[0] for sender in senders for receiver in receivers):
        raise ValueError("candidate senders must precede every receiver")
    methods = [probe.causal.path(senders=senders, receivers=[receiver],
                                sender_tokens="last_prompt", receiver_tokens="last_prompt")
               for receiver in receivers]
    identity = probe.causal.path(senders=[senders[0]], receivers=[receivers[0]],
                                sender_tokens="last_prompt", receiver_tokens="last_prompt")
    reverse_sender = receivers[0][:2]
    reverse_receiver = (*senders[0], "q")
    reverse = probe.causal.path(senders=[reverse_sender], receivers=[reverse_receiver],
                               sender_tokens="last_prompt", receiver_tokens="last_prompt")
    artifact = {
        "schema_version": 1, "model_id": probe.describe()["model_id"],
        "analysis": "selected sender output → receiver Q → answer-token logit margin",
        "selection_strategy": {
            "receiver_pool": "last ceil(10% of decoder layers)",
            "ranking": "descending absolute mean head-to-final-residual effect; layer/head tie-break",
            "sender_pool": "heads strictly before the earliest selected receiver layer",
            "sender_count": len(senders), "receiver_count": len(receivers),
            "validation": "exploratory selection and measurement on the same examples",
        },
        "senders": [list(item) for item in senders],
        "receivers": [list(item) for item in receivers],
        "sender_tokens": "last_prompt", "receiver_tokens": "last_prompt",
        "alignment": "position", "epsilon": epsilon,
        "normalization": "(patched margin - clean margin) / (donor margin - clean margin), then per-pair mean",
        "control_rule": "all other attention head outputs fixed to clean cache; MLP/LayerNorm recompute; fresh pass injects receiver Q only",
        "interpretation": "Measured candidate routes through one query endpoint; not a direct edge, complete circuit, or head-role classification.",
        "official_figure_reproduction": False,
        "edges": [{"sender": list(sender), "receiver": list(receiver), "per_pair": []}
                  for receiver in receivers for sender in senders],
        "pairs": [], "controls": [], "actual_forward_passes": 0,
    }
    tokenizer = getattr(processor, "tokenizer", processor)
    for pair in pairs:
        clean, donor = prepare_inputs(pair, processor, device)
        metric, token_info = first_answer_metric(pair, tokenizer, clean_inputs=clean, donor_inputs=donor)
        artifact["pairs"].append({"pair_id": pair["id"], "clean_answer": pair["clean_answer"],
                                  "donor_answer": pair["donor_answer"], "metric": token_info,
                                  "source": pair.get("source", {})})
        for receiver_index, method in enumerate(methods):
            result = method.sweep(clean, donor=donor, metric=metric, alignment="position")
            values, eligible = normalized_effect(result, epsilon)
            if eligible.numel() != 1:
                raise ValueError("each canonical pair must produce one model example")
            baseline = float(result.tensors["baseline_score"].item())
            donor_margin = float(result.tensors["donor_score"].item())
            artifact["actual_forward_passes"] += result.metadata["actual_forward_passes"]
            for sender_index in range(len(senders)):
                patched = float(result.tensors["intervention_score"][sender_index].item())
                artifact["edges"][receiver_index * len(senders) + sender_index]["per_pair"].append({
                    "pair_id": pair["id"], "clean_margin": baseline, "donor_margin": donor_margin,
                    "patched_margin": patched, "raw_delta": patched - baseline,
                    "normalizer": donor_margin - baseline,
                    "normalized_effect": float(values[sender_index, 0]) if bool(eligible[0]) else None,
                })
        for kind, method, inputs in (("identity_donor", identity, clean),
                                     ("later_sender_to_earlier_receiver_q", reverse, donor)):
            result = method.run(clean, donor=inputs, metric=metric,
                                alignment="strict" if kind == "identity_donor" else "position")
            delta = float(result.tensors["effect"].item())
            artifact["controls"].append({
                "kind": kind, "pair_id": pair["id"], "sender": list(method.senders[0]),
                "receiver": list(method.receivers[0]),
                "clean_margin": float(result.tensors["baseline_score"].item()),
                "patched_margin": float(result.tensors["intervention_score"].item()),
                "raw_delta": delta, "expected_raw_delta": 0.0,
                "absolute_tolerance": 1e-8, "passed": abs(delta) <= 1e-8,
            })
            artifact["actual_forward_passes"] += 4
    for edge in artifact["edges"]:
        edge["mean_normalized_effect"] = aggregate_pair_effects(edge["per_pair"])
    return artifact
