"""Layer/token vocabulary readouts from lens results or precomputed matrices."""
from ..core import ProbeResult
from ._common import array, heatmap, tensor as result_tensor


def plot_lens_heatmap(data, *, tensor=None, target_token_id=None, value_kind="rank",
                      layer_labels=None, token_labels=None, annotations=None,
                      tokenizer=None, head=None, batch_index=0, ax=None, title=None,
                      value_label=None, log_scale=None, cmap=None, vmin=None,
                      vmax=None, colorbar=True):
    """Draw native layer/position readouts without changing token text.

    ``data`` is a ``ProbeResult`` or a matrix of precomputed values [layer,
    position]. Logits [layer, position, vocabulary] require a fixed
    ``target_token_id`` (one ID or one per position), and are converted to
    rank (1 is best), probability, or logit. For Attention Lens logits
    [layer, position, head, vocabulary], select one ``head`` explicitly.

    Packed ``ProbeResult.positions`` coordinates select ``batch_index`` and
    label actual expanded-token positions. The caller may replace labels,
    supply exact decoded annotations, or use ``tokenizer`` for top-1 text.
    Missing matrix measurements (NaN) stay gray; no cells are interpolated.
    Returns a matplotlib Axes. A target subword is not a whole-word score.
    """
    import numpy as np
    if value_kind not in {"rank", "probability", "logit"}:
        raise ValueError("value_kind must be rank, probability, or logit")
    if type(batch_index) is not int or batch_index < 0:
        raise ValueError("batch_index must be a nonnegative integer")
    values, _ = result_tensor(data, tensor, ("ranks", "rank", "logits"))
    if isinstance(data, ProbeResult):
        layer_labels = data.metadata.get("layers") if layer_labels is None else layer_labels
        if "positions" in data.tensors:
            coordinates = array(data.tensors["positions"])
            if coordinates.ndim != 2 or coordinates.shape != (values.shape[1], 2):
                raise ValueError("result positions must be [packed_position,2] and match columns")
            selected = coordinates[:, 0] == batch_index
            if not selected.any():
                raise ValueError("batch_index has no captured positions")
            values = values[:, selected]
            if token_labels is None:
                token_labels = coordinates[selected, 1].astype(int).tolist()
    logits = values.ndim > 2
    if logits:
        if values.ndim == 4:
            if type(head) is not int or not 0 <= head < values.shape[2]:
                raise ValueError("Attention Lens logits require a valid head index")
            values = values[:, :, head, :]
        elif values.ndim != 3 or head is not None:
            raise ValueError("logits must be [layer,position,vocabulary], or select an attention head")
        if not np.isfinite(values).all():
            raise ValueError("logits must be finite")
        targets = np.asarray(target_token_id)
        if target_token_id is None or not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("logits require integer target_token_id")
        if targets.ndim == 0:
            targets = np.full(values.shape[1], int(targets), dtype=int)
        if targets.shape != (values.shape[1],) or (targets < 0).any() or (targets >= values.shape[-1]).any():
            raise ValueError("target IDs must be in vocabulary and match displayed positions")
        chosen = np.take_along_axis(values, targets[None, :, None], axis=-1)[..., 0]
        if tokenizer is not None and annotations is None:
            annotations = [[tokenizer.decode([int(index)]) for index in row] for row in values.argmax(-1)]
        if value_kind == "rank":
            # Strictly higher logits precede the target; ties share its rank.
            matrix = 1 + (values > chosen[..., None]).sum(-1)
            vmax = values.shape[-1] if vmax is None else vmax
        elif value_kind == "probability":
            shifted = values - values.max(-1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(-1, keepdims=True)
            matrix = np.take_along_axis(probabilities, targets[None, :, None], axis=-1)[..., 0]
        else:
            matrix = chosen
    else:
        matrix = values
        if target_token_id is not None or tokenizer is not None or head is not None:
            raise ValueError("target_token_id, tokenizer, and head require vocabulary logits")
    log_scale = value_kind == "rank" if log_scale is None else log_scale
    vmin = 1 if vmin is None and value_kind == "rank" else vmin
    return heatmap(matrix, ax=ax, row_labels=layer_labels, column_labels=token_labels,
                   title=title, log_scale=log_scale, cmap=cmap, vmin=vmin, vmax=vmax,
                   colorbar=colorbar, value_label=value_label or {
                       "rank": "Target token rank (1 is best)", "probability": "Target token probability",
                       "logit": "Target token logit"}[value_kind], annotations=annotations)
