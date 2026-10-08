"""Plotting contracts shared by model-independent visualizations.

Optional plotting packages are imported only when a figure is requested.
"""
from ..core import ProbeResult


def array(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    if np.iscomplexobj(value):
        raise ValueError("plot values must be real numbers")
    try:
        result = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("plot values must be numeric") from error
    if np.isinf(result).any():
        raise ValueError("plot values cannot contain infinity")
    return result


def tensor(value, key, candidates):
    if not isinstance(value, ProbeResult):
        return array(value), None
    key = key or next((name for name in candidates if name in value.tensors), None)
    if key is None or key not in value.tensors:
        raise ValueError(f"result has no requested tensor; available: {list(value.tensors)}")
    return array(value.tensors[key]), key


def pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Install vlm-probing[visualization] to draw figures") from error
    return plt


def axes(ax=None, *, figsize=(9, 4)):
    return ax if ax is not None else pyplot().subplots(figsize=figsize, layout="constrained")[1]


def labels(values, count, name):
    result = list(range(count)) if values is None else list(values)
    if len(result) != count:
        raise ValueError(f"{name} must contain {count} labels")
    return result


def normalization(values, *, signed=False, log_scale=False, vmin=None, vmax=None):
    import numpy as np
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError("plot needs at least one finite measurement")
    if signed and log_scale:
        raise ValueError("a signed scale cannot be logarithmic")
    if signed:
        limit = max(float(np.abs(finite).max()), 1e-12)
        lo, hi = -limit if vmin is None else vmin, limit if vmax is None else vmax
        if not lo < 0 < hi:
            raise ValueError("signed color limits must straddle zero")
        return TwoSlopeNorm(vcenter=0, vmin=lo, vmax=hi)
    lo = float(finite.min()) if vmin is None else vmin
    hi = float(finite.max()) if vmax is None else vmax
    if lo > hi:
        raise ValueError("vmin cannot exceed vmax")
    if log_scale:
        if (finite <= 0).any() or lo <= 0:
            raise ValueError("logarithmic color scales require positive measurements")
        if lo == hi:
            hi = lo * 10
        return LogNorm(lo, hi)
    if lo == hi:
        hi = lo + max(abs(lo), 1) * 1e-6
    return Normalize(lo, hi)


def heatmap(values, *, ax=None, row_labels=None, column_labels=None,
            title=None, xlabel="Token position", ylabel="Layer", signed=False,
            log_scale=False, cmap=None, vmin=None, vmax=None, colorbar=True,
            value_label=None, annotations=None, annotation_size=6):
    import numpy as np
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("heatmap must be a nonempty two-dimensional matrix")
    rows = labels(row_labels, values.shape[0], "row_labels")
    columns = labels(column_labels, values.shape[1], "column_labels")
    if annotations is not None and np.asarray(annotations).shape != values.shape:
        raise ValueError("annotations must match the heatmap shape")
    norm = normalization(values, signed=signed, log_scale=log_scale, vmin=vmin, vmax=vmax)
    ax = axes(ax, figsize=(max(5, min(14, len(columns) * .45)), max(3, len(rows) * .22)))
    palette = pyplot().get_cmap(cmap or ("RdBu_r" if signed else "viridis_r")).with_extremes(bad="#dddddd")
    image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", interpolation="nearest",
                      cmap=palette, norm=norm)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xticks(range(len(columns)), columns, rotation=50, ha="right")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if title:
        ax.set_title(title)
    if annotations is not None:
        for row, column in np.ndindex(values.shape):
            if np.isfinite(values[row, column]):
                ax.text(column, row, str(annotations[row][column]), ha="center", va="center",
                        fontsize=annotation_size, color="#102438",
                        bbox={"facecolor": "white", "alpha": .7, "edgecolor": "none", "pad": .2})
    if colorbar:
        ax.figure.colorbar(image, ax=ax, label=value_label or "Value", shrink=.9)
    return ax
