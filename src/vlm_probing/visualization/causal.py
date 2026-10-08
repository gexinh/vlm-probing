"""Measured interventions and attribution estimates with explicit coordinates."""
from collections.abc import Mapping

from ..core import ProbeResult
from ._common import array, axes, heatmap, pyplot, tensor as result_tensor


def coordinate_grid(coordinates, values, *, row_labels=None, column_labels=None):
    """Place measured [row,column] coordinates in a grid; unmeasured cells are NaN.

    Returns ``(grid, row_labels, column_labels)``. Labels can describe a larger
    searched space, including unmeasured sites, and their given order is kept.
    Duplicate coordinates are rejected rather than averaged implicitly.
    """
    import numpy as np
    if hasattr(coordinates, "detach"):
        coordinates = coordinates.detach().cpu().numpy()
    points = np.asarray(coordinates)
    values = array(values)
    if points.ndim != 2 or points.shape[1] != 2 or values.shape != (len(points),):
        raise ValueError("coordinates must be [measurement,2] with one value per measurement")
    if not np.issubdtype(points.dtype, np.integer) or (points < 0).any():
        raise ValueError("coordinates must be nonnegative integers")
    if not len(points) or len({tuple(point) for point in points}) != len(points):
        raise ValueError("coordinates must be nonempty and unique")
    rows = sorted(set(points[:, 0].tolist())) if row_labels is None else list(row_labels)
    columns = sorted(set(points[:, 1].tolist())) if column_labels is None else list(column_labels)
    if len(set(rows)) != len(rows) or len(set(columns)) != len(columns):
        raise ValueError("coordinate labels must be unique")
    row_indices, column_indices = {x: i for i, x in enumerate(rows)}, {x: i for i, x in enumerate(columns)}
    grid = np.full((len(rows), len(columns)), np.nan)
    for point, value in zip(points, values):
        if point[0] not in row_indices or point[1] not in column_indices:
            raise ValueError("coordinates fall outside the supplied labels")
        grid[row_indices[point[0]], column_indices[point[1]]] = value
    return grid, rows, columns


def plot_causal_heatmap(data, *, tensor=None, layer_labels=None, column_labels=None,
                        batch_index=0, ax=None, title=None, value_label="Score change",
                        signed=True, cmap=None, vmin=None, vmax=None, colorbar=True,
                        highlight_count=0):
    """Plot layer/token or layer/head measurements, with a zero-centered scale.

    A precomputed [layer,column] matrix is used verbatim. ``ProbeResult``
    path sweeps are laid out using their actual ``senders`` [layer,head]
    coordinates. Other results support ``effect`` [layer,batch],
    ``estimated_effect`` [layer], and ``token_scores`` [layer,batch,token].
    Use ``tensor='token_scores'`` to show attribution at individual tokens.
    Missing/unsearched cells remain gray. No circuit or head role is inferred.
    """
    import numpy as np
    from matplotlib.patches import Rectangle
    if type(highlight_count) is not int or highlight_count < 0:
        raise ValueError("highlight_count must be a nonnegative integer")
    if type(batch_index) is not int or batch_index < 0:
        raise ValueError("batch_index must be a nonnegative integer")
    values, key = result_tensor(data, tensor, ("effect", "estimated_effect", "token_scores"))
    xlabel = "Position"
    if isinstance(data, ProbeResult):
        if "senders" in data.tensors:
            if values.ndim == 2:
                if batch_index >= values.shape[1]:
                    raise ValueError("batch_index exceeds the path result batch dimension")
                values = values[:, batch_index]
            values, layer_labels, column_labels = coordinate_grid(
                data.tensors["senders"], values, row_labels=layer_labels, column_labels=column_labels)
            xlabel = "Attention head"
        else:
            layer_labels = data.metadata.get("layers") if layer_labels is None else layer_labels
            if key == "token_scores":
                if values.ndim != 3 or batch_index >= values.shape[1]:
                    raise ValueError("token_scores must be [layer,batch,token]")
                values = values[:, batch_index]
            elif values.ndim == 2:
                if batch_index >= values.shape[1]:
                    raise ValueError("batch_index exceeds the effect batch dimension")
                values = values[:, batch_index, None]
                column_labels = ["Effect"] if column_labels is None else column_labels
    if values.ndim == 1:
        values = values[:, None]
    ax = heatmap(values, ax=ax, row_labels=layer_labels, column_labels=column_labels,
                 xlabel=xlabel, title=title, signed=signed, cmap=cmap, vmin=vmin,
                 vmax=vmax, colorbar=colorbar, value_label=value_label)
    finite = [(r, c) for r, c in np.ndindex(values.shape) if np.isfinite(values[r, c])]
    finite.sort(key=lambda point: (-abs(values[point]), *point))
    for row, column in finite[:highlight_count]:
        ax.add_patch(Rectangle((column - .5, row - .5), 1, 1, fill=False,
                               edgecolor="#EDAF00", linewidth=1.7))
    return ax


def plot_intervention_curves(curves, *, x=None, baseline=None, relative=False,
                              ax=None, title=None, xlabel="Layer", ylabel="Score",
                              colors=None, markers=None, tensor=None, batch_index=0):
    """Plot named measured curves with unchanged x coordinates.

    Each value is a vector, ``{'x': ..., 'y': ...}``, or ``ProbeResult``.
    Results default to ``intervention_score`` (then effect/estimated_effect).
    Result layer metadata supplies x coordinates. If ``relative=True``,
    plot ``(y-baseline)/baseline``; a baseline is required and cannot be zero.
    Missing values remain gaps. The function does not smooth or refit curves.
    Baseline can be one number, a vector, or a name-to-baseline mapping.
    """
    import numpy as np
    if not isinstance(curves, Mapping) or not curves:
        raise ValueError("curves must be a nonempty mapping")
    if type(batch_index) is not int or batch_index < 0:
        raise ValueError("batch_index must be a nonnegative integer")
    if relative and baseline is None:
        raise ValueError("relative curves require a nonzero baseline")
    prepared = []
    for name, curve in curves.items():
        coordinates = x
        if isinstance(curve, Mapping):
            if "y" not in curve:
                raise ValueError("curve dictionaries require y")
            values = array(curve["y"])
            coordinates = curve.get("x", x)
        else:
            values, _ = result_tensor(curve, tensor, ("intervention_score", "effect", "estimated_effect"))
            if isinstance(curve, ProbeResult):
                if values.ndim == 2:
                    if batch_index >= values.shape[1]:
                        raise ValueError("batch_index exceeds the curve batch dimension")
                    values = values[:, batch_index]
                coordinates = curve.metadata.get("layers") if x is None else x
        if values.ndim != 1 or not len(values) or not np.isfinite(values).any():
            raise ValueError("each curve needs a nonempty vector with a finite measurement")
        coordinates = np.arange(len(values)) if coordinates is None else array(coordinates)
        if coordinates.shape != values.shape or not np.isfinite(coordinates).all():
            raise ValueError("x coordinates must be finite and match curve values")
        base = baseline.get(name) if isinstance(baseline, Mapping) else baseline
        if relative:
            base = array(base)
            if not np.isfinite(base).all() or (base == 0).any() or base.shape not in ((), values.shape):
                raise ValueError("relative baseline must be finite, nonzero, and scalar or match the curve")
            values = (values - base) / base
        prepared.append((name, coordinates, values))
    ax = axes(ax)
    for name, coordinates, values in prepared:
        ax.plot(coordinates, values, label=name,
                color=colors.get(name) if colors else None,
                marker=markers.get(name) if markers else None)
    if relative:
        ax.axhline(0, linestyle="--", color="#8393a5", linewidth=.8)
    elif baseline is not None and not isinstance(baseline, Mapping) and array(baseline).ndim == 0:
        if not np.isfinite(float(baseline)):
            raise ValueError("baseline must be finite")
        ax.axhline(float(baseline), linestyle="--", color="#8393a5", linewidth=.8, label="Baseline")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=.2)
    ax.legend()
    return ax


def plot_connections(edges, *, node_layers, top_k=20, ax=None, title=None,
                     value_label="Edge score", colorbar=True):
    """Draw supplied signed edges arranged by model layer; infer no connections.

    ``edges`` contains ``(source_name, receiver_name, score)`` triples;
    ``node_layers`` maps every node name to its numeric model-layer coordinate.
    Only the largest ``top_k`` absolute finite scores are shown. Width indicates
    absolute score and red/blue indicates sign. Attribution edges are estimates,
    not causal measurements unless the supplied scores are intervention results.
    """
    import math
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.cm import ScalarMappable
    if not isinstance(node_layers, Mapping) or not node_layers:
        raise ValueError("node_layers must map node names to numeric layer coordinates")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    selected, seen = [], set()
    for edge in edges:
        if len(edge) != 3:
            raise ValueError("edges must be (source,receiver,score) triples")
        sender, receiver, score = edge
        if sender not in node_layers or receiver not in node_layers or not math.isfinite(float(score)):
            raise ValueError("edge endpoints require layer coordinates and finite scores")
        if (sender, receiver) in seen:
            raise ValueError("duplicate edge")
        seen.add((sender, receiver))
        selected.append((sender, receiver, float(score)))
    if not selected:
        raise ValueError("connections need at least one measured or estimated edge")
    selected = sorted(selected, key=lambda edge: -abs(edge[2]))[:top_k]
    nodes = {node for sender, receiver, _ in selected for node in (sender, receiver)}
    grouped = {}
    for node in sorted(nodes):
        layer = node_layers[node]
        from numbers import Real
        if not isinstance(layer, Real) or not math.isfinite(layer):
            raise ValueError("node layers must be finite numbers")
        grouped.setdefault(layer, []).append(node)
    positions = {node: (layer, index - (len(group) - 1) / 2)
                 for layer, group in grouped.items() for index, node in enumerate(group)}
    limit = max(abs(score) for _, _, score in selected) or 1e-12
    norm = TwoSlopeNorm(0, -limit, limit)
    palette = pyplot().get_cmap("RdBu_r")
    ax = axes(ax, figsize=(11, 5))
    for sender, receiver, score in selected:
        ax.add_patch(FancyArrowPatch(positions[sender], positions[receiver], arrowstyle="-|>",
                                     mutation_scale=11, shrinkA=10, shrinkB=10,
                                     connectionstyle="arc3,rad=.08", color=palette(norm(score)),
                                     linewidth=.7 + 3 * abs(score) / limit, alpha=.85))
    for node, (layer, row) in positions.items():
        ax.scatter([layer], [row], s=140, facecolor="#eef3fa", edgecolor="#344b68", zorder=3)
        ax.annotate(str(node), (layer, row), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_xticks(sorted(grouped))
    ax.set_xlabel("Model layer")
    ax.set_yticks([])
    ax.margins(x=.12, y=.25)
    if title:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(ScalarMappable(norm=norm, cmap=palette), ax=ax, label=value_label)
    return ax
