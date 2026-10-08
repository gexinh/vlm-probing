"""Diagrams of independently measured head paths and receiver interventions.

These functions draw tested intervention routes; they do not discover circuits
or infer semantic head roles. Matplotlib is imported only when drawing.
"""
import math
import torch

def _validated_paths(senders, effects):
    senders = [tuple(sender) for sender in senders]
    values = torch.as_tensor(effects).detach().float().cpu()
    if values.ndim == 1:
        values = values[:, None]
    if (values.ndim != 2 or values.shape[0] != len(senders) or len(set(senders)) != len(senders)
            or any(len(sender) != 2 or any(type(coordinate) is not int or coordinate < 0
                                         for coordinate in sender) for sender in senders)):
        raise ValueError("effects must match unique nonnegative (layer, head) sender coordinates")
    if torch.isinf(values).any() or not torch.isfinite(values).any():
        raise ValueError("path diagrams require finite measurements and no infinite values")
    return senders, values

def plot_head_paths(senders, effects, *, pairs=None, model_name=None, top_k=5):
    """Draw independently measured sender-head paths to the final readout.

    Heads are ranked by the absolute mean of their eligible per-pair normalized
    effects. The diagram displays all per-pair values, including sign changes.
    Arrows are intervention routes through the controlled run, not recovered
    head-to-head connections or an assertion about functional head classes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    senders, values = _validated_paths(senders, effects)
    if pairs is not None and len(pairs) != values.shape[1]:
        raise ValueError("one pair record is required per effect column")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    means = values.nanmean(dim=1)
    ranked = sorted((i for i in range(len(senders)) if torch.isfinite(means[i])),
                    key=lambda i: (-abs(means[i].item()), senders[i]))[:top_k]
    limit = max(abs(means[index].item()) for index in ranked) or 1e-6
    norm = TwoSlopeNorm(0, -limit, limit)
    cmap = plt.get_cmap("RdBu")
    figure, axis = plt.subplots(figsize=(14, max(6.7, len(ranked) * .82 + 2.8)))
    axis.set(xlim=(0, 15), ylim=(0, 10))
    axis.axis("off")
    subtitle = "Measured head-to-readout paths · independent interventions · not a recovered circuit"
    figure.suptitle(f"{model_name + ' | ' if model_name else ''}IOI-style path patching", fontsize=16, y=.97)
    axis.text(7.5, 9.65, subtitle, ha="center", fontsize=11, color="#526476")

    def box(x, y, width, height, text, face="#eff4f8", edge="#899cad", fontsize=10):
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.10,rounding_size=0.18",
                               facecolor=face, edgecolor=edge, linewidth=1.15)
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)

    ys = torch.linspace(8.35, 3.15, len(ranked)).tolist() if len(ranked) > 1 else [5.8]
    heights = min(.76, 4.4 / max(len(ranked), 1))
    for rank, (index, y) in enumerate(zip(ranked, ys), 1):
        layer, head = senders[index]
        color = cmap(norm(means[index].item()))
        pale = tuple(.83 + .17 * channel for channel in color[:3])
        measurements = "  ·  ".join(
            f"E{pair_index + 1}: {value:+.3f}" if torch.isfinite(value) else f"E{pair_index + 1}: undefined"
            for pair_index, value in enumerate(values[index]))
        label = f"Head L{layer}.H{head}  |  mean Φ = {means[index].item():+.3f}\n{measurements}"
        box(.30, y - heights / 2, 4.0, heights, label, face=pale, edge=color, fontsize=9.5)
        arrow = FancyArrowPatch((4.44, y), (5.20, 6.04), arrowstyle="-|>",
                                connectionstyle="arc3,rad=0.0", mutation_scale=12,
                                color=color, linewidth=1.2 + 2.5 * abs(means[index].item()) / limit)
        axis.add_patch(arrow)
    axis.text(2.3, 8.96, "Donor head output replaces the clean value\nat the last prompt token", ha="center", fontsize=10)
    box(5.34, 4.75, 3.55, 2.3, "Controlled propagation\n\nOther head outputs: clean and frozen\nMLP / LayerNorm: recomputed", fontsize=10)
    box(9.55, 5.05, 2.14, 1.75, "Receiver\nFinal residual\nLast prompt token", face="#e7f1f4")
    box(12.37, 5.05, 2.13, 1.75, "Fresh clean run\nInject receiver\nNorm → LM head", face="#f8f0e4")
    for start, end in (((9.04, 5.90), (9.42, 5.90)), ((11.82, 5.90), (12.24, 5.90))):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                      linewidth=1.6, color="#516372"))
    axis.text(10.65, 4.20, "Cache the controlled receiver, then inject only that receiver\ninto a fresh clean forward pass; the freezing hooks are removed.",
              ha="center", fontsize=9.5, color="#526476")
    axis.text(7.5, 2.25,
              r"$d=\mathrm{logit}(y_{clean,1})-\mathrm{logit}(y_{donor,1})$"
              "      " + r"$\phi_i=(d_{patched,i}-d_{clean,i})/(d_{donor,i}-d_{clean,i})$"
              "      " + r"$\Phi=\mathrm{mean}_i\,\phi_i$",
              ha="center", fontsize=12)
    legend = "Blue / Φ > 0: shift toward the donor model's margin.  Red / Φ < 0: shift in the opposite direction."
    axis.text(7.5, 1.75, legend, ha="center", fontsize=10, color="#344856")
    axis.text(7.5, 1.32, "Line width encodes |mean Φ|. Every head is tested separately; effects are not summed.",
              ha="center", fontsize=10, color="#526476")
    if pairs:
        labels = [f"E{index + 1}: expected {pair['clean_answer']} → {pair['donor_answer']}"
                  for index, pair in enumerate(pairs)]
        axis.text(7.5, .87, "    |    ".join(labels), ha="center", fontsize=10)
    axis.text(7.5, .42,
              "Candidate labels are ground truth. A donor input need not move the model toward its expected answer.\n"
              "Head indices are zero-based. No head roles or unmeasured head-to-head edges are assigned.",
              ha="center", fontsize=9.5, color="#526476")
    figure.subplots_adjust(top=.91, bottom=.02, left=.015, right=.985)
    return figure

def plot_receiver_paths(artifact, *, model_name=None):
    """Draw separately measured sender-output → receiver-Q/K/V → readout paths.

    The artifact needs ``edges`` with unique ``sender=[layer,head]``,
    ``receiver=[layer,head,component]`` and ``per_pair`` measurements containing
    ``id`` (or ``pair_id``) and ``normalized_effect``. A supplied mean must
    agree with the mean of the finite per-pair measurements. No edge is invented.
    Downstream readout arrows illustrate the ordinary fresh forward computation;
    their causal contribution has not been measured separately.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    edges = artifact.get("edges", [])
    if not edges:
        raise ValueError("receiver-path diagrams need measured edges")
    seen, measured, pair_ids = set(), [], []
    for edge in edges:
        sender, receiver = tuple(edge.get("sender", ())), tuple(edge.get("receiver", ()))
        if (len(sender) != 2 or len(receiver) != 3 or receiver[2] not in ("q", "k", "v")
                or any(type(value) is not int or value < 0 for value in (*sender, *receiver[:2]))):
            raise ValueError("edges require sender [layer,head] and receiver [layer,head,q/k/v]")
        key = (sender, receiver)
        if key in seen:
            raise ValueError("duplicate measured edge")
        seen.add(key)
        per_pair = edge.get("per_pair", [])
        if not per_pair or any(not isinstance(item.get("id", item.get("pair_id")), str)
                               or not item.get("id", item.get("pair_id"))
                               for item in per_pair):
            raise ValueError("each edge needs identified per-pair measurements")
        current_ids = [item.get("id", item.get("pair_id")) for item in per_pair]
        if len(set(current_ids)) != len(current_ids):
            raise ValueError("duplicate pair ID in edge measurements")
        if not pair_ids:
            pair_ids = current_ids
        if current_ids != pair_ids:
            raise ValueError("edge measurements must use the same pair order")
        values = [float(item["normalized_effect"]) if item.get("normalized_effect") is not None
                  else math.nan for item in per_pair]
        if any(math.isinf(value) for value in values):
            raise ValueError("infinite receiver-path effects are invalid")
        eligible = [value for value in values if math.isfinite(value)]
        mean = sum(eligible) / len(eligible) if eligible else math.nan
        supplied = edge.get("mean_normalized_effect")
        if supplied is not None and (not math.isfinite(float(supplied)) or not math.isfinite(mean)
                                     or not math.isclose(float(supplied), mean, rel_tol=1e-5, abs_tol=1e-6)):
            raise ValueError("edge mean disagrees with its per-pair measurements")
        measured.append((sender, receiver, values, mean))
    finite = [abs(mean) for _, _, _, mean in measured if math.isfinite(mean)]
    if not finite:
        raise ValueError("receiver-path diagrams need at least one finite measured mean")
    senders = sorted({sender for sender, _, _, _ in measured})
    receivers = sorted({receiver for _, receiver, _, _ in measured})
    scale = max(finite) or 1e-6
    norm, cmap = TwoSlopeNorm(0, -scale, scale), plt.get_cmap("RdBu")
    height = max(9.3, .31 * len(measured) + 6.3)
    figure = plt.figure(figsize=(15.5, height))
    axis = figure.add_axes([.035, .445, .93, .43])
    axis.set(xlim=(0, 15.5), ylim=(0, 10))
    axis.axis("off")
    title = model_name or artifact.get("model_id") or "Model"
    figure.suptitle(f"{title} | measured sender → receiver paths", fontsize=16, y=.978)
    figure.text(.5, .940,
                "Each colored arrow is an independently tested sender-output → receiver-component → answer-margin route.",
                ha="center", fontsize=11, color="#526476")
    cases = {item["pair_id"]: item for item in artifact.get("pairs", [])}
    if all(pair_id in cases for pair_id in pair_ids):
        context = "   |   ".join(
            f"E{index + 1}: {cases[pair_id]['clean_answer']} → {cases[pair_id]['donor_answer']}"
            for index, pair_id in enumerate(pair_ids)
        )
        figure.text(.5, .905, f"Expected answers, clean → donor: {context}",
                    ha="center", fontsize=10, color="#344856")

    def box(x, y, width, height, text, face="#eff4f8", fontsize=10):
        axis.add_patch(FancyBboxPatch((x, y - height / 2), width, height,
                                     boxstyle="round,pad=0.10,rounding_size=0.16",
                                     facecolor=face, edgecolor="#899cad", linewidth=1.15))
        axis.text(x + width / 2, y, text, ha="center", va="center", fontsize=fontsize)

    sender_ys = torch.linspace(8.15, 2.15, len(senders)).tolist() if len(senders) > 1 else [5.15]
    receiver_ys = torch.linspace(7.45, 3.0, len(receivers)).tolist() if len(receivers) > 1 else [5.15]
    sender_y, receiver_y = dict(zip(senders, sender_ys)), dict(zip(receivers, receiver_ys))
    axis.text(1.8, 9.5, "Donor sender-head output", ha="center", fontsize=11, fontweight="bold")
    axis.text(8.72, 9.5, "Controlled receiver activation", ha="center", fontsize=11, fontweight="bold")
    for sender, y in sender_y.items():
        box(.2, y, 3.15, min(1.05, 5.2 / len(senders)),
            f"Head L{sender[0]}.H{sender[1]}\nLast prompt token")
    for receiver, y in receiver_y.items():
        box(7.5, y, 2.5, min(1.25, 5.3 / len(receivers)),
            f"Head L{receiver[0]}.H{receiver[1]}\nReceiver {receiver[2].upper()}\nLast prompt token",
            face="#e4f1f4", fontsize=10)
    for receiver in receivers:
        incoming = sorted((entry for entry in measured if entry[1] == receiver), key=lambda entry: entry[0])
        offsets = torch.linspace(.43, -.43, len(incoming)).tolist() if len(incoming) > 1 else [0.]
        for (sender, _, _, mean), offset in zip(incoming, offsets):
            color = cmap(norm(mean)) if math.isfinite(mean) and mean != 0 else "#bbc2c8"
            dashed = not math.isfinite(mean) or mean == 0
            y0, y1 = sender_y[sender], receiver_y[receiver] + offset
            axis.add_patch(FancyArrowPatch((3.47, y0), (7.36, y1), arrowstyle="-|>",
                                          mutation_scale=13, connectionstyle="arc3,rad=0.0",
                                          color=color, linewidth=1.0 + (2.8 * abs(mean) / scale if math.isfinite(mean) else 0),
                                          linestyle=":" if dashed else "-"))
            label = ("Φ = 0" if mean == 0 else f"{mean:+.3f}") if math.isfinite(mean) else "undefined"
            label_color = "#145073" if mean > 0 else "#8a2335" if mean < 0 else "#687781"
            axis.text(6.7, y1, label,
                      ha="center", va="center", fontsize=8.5, color=label_color,
                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": .96})
    box(10.7, 5.15, 2.4, 1.85, "Separate clean replay\nInject one receiver\nDownstream heads live", face="#f8f0e4", fontsize=10)
    box(13.72, 5.15, 1.42, 1.85, "Answer\nfirst-token\nlogit margin", face="#f1f4e9", fontsize=9.5)
    for receiver, y in receiver_y.items():
        axis.add_patch(FancyArrowPatch((10.13, y), (10.57, 5.15), arrowstyle="-|>",
                                      mutation_scale=13, color="#8c969f", linewidth=1.3,
                                      linestyle="--"))
    axis.add_patch(FancyArrowPatch((13.23, 5.15), (13.59, 5.15), arrowstyle="-|>",
                                  mutation_scale=13, color="#8c969f", linewidth=1.3, linestyle="--"))
    axis.text(12.45, 3.5, "Dashed gray: normal downstream computation\n(not separately tested)",
              ha="center", fontsize=9, color="#687781")
    axis.text(7.8, .80,
              "During receiver caching: all other attention-head outputs are frozen to clean; MLP / LayerNorm recompute.\n"
              "For scoring: remove those hooks and inject only the receiver into a fresh clean forward pass.",
              ha="center", fontsize=10, color="#526476")

    table_axis = figure.add_axes([.16, .155, .68, .245])
    table_axis.axis("off")
    rows = []
    for sender, receiver, values, mean in sorted(measured, key=lambda entry: (entry[0], entry[1])):
        rows.append([f"L{sender[0]}.H{sender[1]} → L{receiver[0]}.H{receiver[1]} {receiver[2].upper()}",
                     *[f"{value:+.4f}" if math.isfinite(value) else "undefined" for value in values],
                     f"{mean:+.4f}" if math.isfinite(mean) else "undefined"])
    table = table_axis.table(cellText=rows,
                             colLabels=["Measured path", *[f"Example {i + 1} φ" for i in range(len(pair_ids))], "Mean Φ"],
                             cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.3)
    table.scale(1., 1.4)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d3dce3")
        if row == 0:
            cell.set_facecolor("#e8eef3")
            cell.set_text_props(weight="bold")
        elif row % 2:
            cell.set_facecolor("#f6f8fa")
    figure.text(.5, .104,
                "Edges show mean Φ: blue toward donor margin; red opposite; line width |mean Φ|. Zero means: dotted light gray.",
                ha="center", fontsize=10, color="#344856")
    figure.text(.5, .063,
                "Gray downstream arrows describe normal computation and are not separately measured causal edges.\n"
                "This diagram contains measured candidate paths, not a recovered circuit or functional head-role classification.",
                ha="center", fontsize=10, color="#526476")
    return figure
