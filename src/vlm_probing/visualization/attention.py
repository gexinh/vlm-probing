"""Overlay caller-aligned visual-token maps without guessing image geometry."""
from ..core import ProbeResult
from ._common import array, axes, normalization, pyplot, tensor as result_tensor


def _indices(selection, count, name):
    import numpy as np
    if hasattr(selection, "detach"):
        selection = selection.detach().cpu().numpy()
    value = np.asarray(selection)
    if value.dtype == bool:
        if value.shape != (count,):
            raise ValueError(f"{name} boolean mask must match the token axis")
        value = np.flatnonzero(value)
    if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{name} must contain integer positions or a boolean mask")
    value = np.where(value < 0, value + count, value)
    if not value.size or (value < 0).any() or (value >= count).any() or len(set(value.tolist())) != len(value):
        raise ValueError(f"{name} must contain nonempty unique positions inside the token axis")
    return value


def _grid_shape(shape):
    if shape is None:
        return None
    shape = tuple(shape)
    if len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("grid_shape must be two positive integers")
    return shape


def _visual_map(data, *, tensor=None, grid_shape=None, visual_positions=None,
                query_positions=None, layer=-1, batch_index=0):
    import numpy as np
    values, key = result_tensor(data, tensor, ("map", "cam", "token_attribution", "rollout",
                                             "patch_relevance", "relevance", "attention"))
    shape = _grid_shape(grid_shape)
    if isinstance(data, ProbeResult):
        if type(batch_index) is not int or batch_index < 0:
            raise ValueError("batch_index must be a nonnegative integer")
        if key == "attention":  # [layer,batch,head,query,key], or one layer.
            if values.ndim == 5:
                values = values[layer]
            if values.ndim != 4 or batch_index >= values.shape[0]:
                raise ValueError("attention result must be [batch,head,query,key], optionally with layers")
            values = values[batch_index].mean(0)
        elif values.ndim in (3, 4):  # [batch,query,key], optionally with layers.
            if values.ndim == 4:
                values = values[layer]
            if batch_index >= values.shape[0]:
                raise ValueError("batch_index exceeds captured map batches")
            values = values[batch_index]
        elif key == "relevance" and values.ndim == 2 and shape is not None and values.shape[1] == shape[0] * shape[1]:
            if batch_index >= values.shape[0]:
                raise ValueError("batch_index exceeds captured relevance batches")
            values = values[batch_index]
        if query_positions is None and "query_mask" in data.tensors:
            query_positions = array(data.tensors["query_mask"])[batch_index].astype(bool)
    if visual_positions is not None or query_positions is not None:
        if values.ndim != 2 or visual_positions is None or query_positions is None:
            raise ValueError("query-key maps require both visual_positions and query_positions")
        query = _indices(query_positions, values.shape[0], "query_positions")
        visual = _indices(visual_positions, values.shape[1], "visual_positions")
        # Mean over only the explicit target query rows; preserve signed values.
        values = values[np.ix_(query, visual)].mean(0)
    if values.ndim == 1:
        if shape is None:
            raise ValueError("flat visual maps require explicit grid_shape; no square-grid guessing")
        if values.size != shape[0] * shape[1]:
            raise ValueError("grid_shape does not match the visual-token count")
        values = values.reshape(shape)
    elif values.ndim == 2:
        if shape is not None and values.shape != shape:
            raise ValueError("two-dimensional visual map must already match grid_shape")
    else:
        raise ValueError("select a spatial map or explicit query/key coordinates before plotting")
    if not np.isfinite(values).any():
        raise ValueError("visual map needs a finite measurement")
    return values


def _image(image):
    import numpy as np
    # PIL image conversion is optional; neither caller image nor arrays mutate.
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    values = np.array(image, copy=True)
    if values.ndim not in (2, 3) or (values.ndim == 3 and values.shape[-1] not in (3, 4)):
        raise ValueError("image must be a PIL image or HxW / HxWx3 / HxWx4 array")
    if min(values.shape[:2]) == 0:
        raise ValueError("image cannot be empty")
    if np.issubdtype(values.dtype, np.floating) and (not np.isfinite(values).all() or values.min() < 0 or values.max() > 1):
        raise ValueError("floating image values must be finite and in [0,1]")
    return values


def _overlay(ax, image, values, *, norm, signed, cmap, alpha, bounding_boxes,
             title, colorbar, value_label):
    import numpy as np
    from matplotlib.patches import Rectangle
    palette = pyplot().get_cmap(cmap or ("RdBu_r" if signed else "Blues")).with_extremes(bad=(0, 0, 0, 0))
    height, width = image.shape[:2]
    extent = (-.5, width - .5, height - .5, -.5)
    ax.imshow(image, extent=extent)
    artist = ax.imshow(np.ma.masked_invalid(values), extent=extent, norm=norm, cmap=palette,
                       interpolation="bilinear", alpha=alpha)
    for box in bounding_boxes or []:
        if len(box) != 4 or not np.isfinite(box).all() or min(box[2:]) <= 0:
            raise ValueError("bounding_boxes must be (x,y,width,height) in the supplied image coordinates")
        ax.add_patch(Rectangle(box[:2], box[2], box[3], fill=False, edgecolor="#20e060", linewidth=1.6))
    ax.set(xlim=(-.5, width - .5), ylim=(height - .5, -.5))
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(artist, ax=ax, shrink=.75, label=value_label)
    return artist


def plot_attention_overlay(image, values, *, tensor=None, grid_shape=None,
                           visual_positions=None, query_positions=None, layer=-1,
                           batch_index=0, signed=False, alpha=.72, cmap=None,
                           vmin=None, vmax=None, bounding_boxes=None, ax=None,
                           title=None, colorbar=True, value_label="Attribution"):
    """Overlay a map on the *actual image represented by its visual grid*.

    Supply the processed crop/padded image used by the model. Flat maps need
    ``grid_shape``; multi-image, tiled, or video inputs must be split and
    aligned by the caller. Query/key results require explicit native
    ``visual_positions`` and ``query_positions`` (or a captured query_mask).
    Heads are averaged for raw attention. ``layer`` indexes captured layers,
    not an assumed global decoder index. No normalization changes the data;
    color limits only control display. Returns a matplotlib Axes.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    image = _image(image)
    values = _visual_map(values, tensor=tensor, grid_shape=grid_shape,
                         visual_positions=visual_positions, query_positions=query_positions,
                         layer=layer, batch_index=batch_index)
    if not signed and (values[values == values] < 0).any():
        raise ValueError("negative visual scores require signed=True")
    norm = normalization(values, signed=signed, vmin=0 if vmin is None and not signed else vmin, vmax=vmax)
    ax = axes(ax, figsize=(6, 5))
    _overlay(ax, image, values, norm=norm, signed=signed, cmap=cmap, alpha=alpha,
             bounding_boxes=bounding_boxes, title=title, colorbar=colorbar, value_label=value_label)
    return ax


def plot_attention_comparison(image, named_maps, *, grid_shape=None,
                              visual_positions=None, query_positions=None, layer=-1,
                              batch_index=0, columns=4, signed=False, shared_scale=False,
                              cmap=None, alpha=.72, axes=None, colorbar=True,
                              value_label="Attribution", titles=None, bounding_boxes=None):
    """Compare named maps with a common image, geometry, and explicit scale.

    ``signed`` can be one boolean or a mapping from method name to boolean.
    Per-map color limits are the default because different methods have
    different units. ``shared_scale=True`` uses one colorbar/scale per
    signedness group; compare magnitudes only when scores share units.
    Returns ``(figure, axes)`` with a two-dimensional axes array.
    """
    import numpy as np
    from collections.abc import Mapping
    if not isinstance(named_maps, Mapping) or not named_maps:
        raise ValueError("named_maps must be a nonempty mapping")
    if type(columns) is not int or columns <= 0:
        raise ValueError("columns must be a positive integer")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    image = _image(image)
    maps = {name: _visual_map(value, grid_shape=grid_shape, visual_positions=visual_positions,
                            query_positions=query_positions, layer=layer, batch_index=batch_index)
            for name, value in named_maps.items()}
    signs = {name: bool(signed[name]) if isinstance(signed, Mapping) else bool(signed) for name in maps}
    if shared_scale and isinstance(cmap, Mapping):
        for sign in set(signs.values()):
            if len({cmap.get(name) for name in maps if signs[name] == sign}) > 1:
                raise ValueError("a shared colorbar requires one colormap per signedness group")
    for name, values in maps.items():
        if not signs[name] and (values[np.isfinite(values)] < 0).any():
            raise ValueError(f"negative scores for {name!r} require signed=True")
    columns = min(columns, len(maps))
    rows = (len(maps) + columns - 1) // columns
    if axes is None:
        figure, axes = pyplot().subplots(rows, columns, squeeze=False,
                                        figsize=(3.5 * columns, 3.3 * rows), layout="constrained")
    else:
        axes = np.asarray(axes, dtype=object)
        if axes.ndim != 2 or axes.size < len(maps):
            raise ValueError("axes must be two-dimensional with one slot per map")
        figure = axes.flat[0].figure
        if any(axis.figure is not figure for axis in axes.flat):
            raise ValueError("comparison axes must share one figure")
    groups = {}
    for name, values in maps.items():
        groups.setdefault(signs[name], []).append(values.flatten())
    norms = {sign: normalization(np.concatenate(values), signed=sign, vmin=None if sign else 0)
             for sign, values in groups.items()} if shared_scale else {}
    artists = {}
    group_axes = {}
    for ax, (name, values) in zip(axes.flat, maps.items()):
        sign = signs[name]
        norm = norms[sign] if shared_scale else normalization(values, signed=sign, vmin=None if sign else 0)
        artist = _overlay(ax, image, values, norm=norm, signed=sign,
                          cmap=cmap.get(name) if isinstance(cmap, Mapping) else cmap,
                          alpha=alpha, bounding_boxes=bounding_boxes,
                          title=titles.get(name, name) if titles else name,
                          colorbar=colorbar and not shared_scale, value_label=value_label)
        artists[sign] = artist
        group_axes.setdefault(sign, []).append(ax)
    for ax in list(axes.flat)[len(maps):]:
        ax.axis("off")
    if colorbar and shared_scale:
        for sign, artist in artists.items():
            figure.colorbar(artist, ax=group_axes[sign], shrink=.75, label=value_label)
    return figure, axes


def plot_input(image, *, prompt=None, answer=None, bounding_boxes=None, ax=None, title=None):
    """Show an input image, optional question/answer, and image-space boxes."""
    import textwrap
    import numpy as np
    from matplotlib.patches import Rectangle
    image = _image(image)
    ax = axes(ax, figsize=(7, 5))
    ax.imshow(image)
    for box in bounding_boxes or []:
        if len(box) != 4 or not np.isfinite(box).all() or min(box[2:]) <= 0:
            raise ValueError("bounding_boxes must be (x,y,width,height)")
        ax.add_patch(Rectangle(box[:2], box[2], box[3], fill=False, edgecolor="#20e060", linewidth=1.6))
    heading = "\n".join(part for part in (title, textwrap.fill(prompt, 75) if prompt else None,
                                        f"Answer: {answer}" if answer is not None else None) if part)
    if heading:
        ax.set_title(heading)
    ax.axis("off")
    return ax
