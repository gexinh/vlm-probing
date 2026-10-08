"""Small-dataset calibration through the public model-bound lens interface."""
from pathlib import Path

from torch.utils.data import DataLoader, Dataset

from ..api.common import selected
from .distribution import train_distribution_lens


class _Readouts(Dataset):
    def __init__(self, activations, teachers):
        self.activations, self.teachers = activations, teachers

    def __len__(self):
        return len(self.teachers)

    def __getitem__(self, index):
        return {"activations": self.activations[index], "teacher_logits": self.teachers[index]}


def fit_model_bound_lens(method, train_inputs, validation_inputs, directory, *,
                         batch_size=64, initial_unembedding=None, initial_bias=None,
                         **training_options):
    """Capture small, independent input iterables once, then optimize decoders.

    Cached tensors live in CPU memory. For datasets too large for memory,
    use ``train_distribution_lens`` with disk-backed cached DataLoaders.
    The caller must supply genuinely disjoint training/validation examples
    and immutable model, tokenizer and dataset identities in ``binding``.
    """
    import torch

    if torch.is_inference_mode_enabled():
        raise ValueError("calibration cannot run in torch.inference_mode()")
    if method.name not in {"tuned", "attention"}:
        raise ValueError("fit_batches supports Tuned and Attention Lens; use Jacobian calibration separately")
    if batch_size < 1 or train_inputs is validation_inputs:
        raise ValueError("positive batch_size and separate training/validation inputs are required")
    initialize = initial_unembedding is not None or initial_bias is not None
    if initialize and (method.name != "attention" or method.kernels):
        raise ValueError("head initialization requires a new Attention Lens")
    if any(value == "unspecified" for layer in method.layers for value in method._binding(layer).values()):
        raise ValueError("fit_batches requires explicit model/tokenizer/calibration identities via binding")

    def capture(inputs):
        activations = {layer: [] for layer in method.layers}
        teachers, references = [], {}
        for batch in inputs:
            trace, _, mask, values = method._capture(batch)
            teachers.append(selected(trace.logits, mask).detach().cpu())
            for layer, hidden in zip(method.layers, values):
                references[layer] = (hidden.device, hidden.dtype)
                activations[layer].append(selected(hidden, mask).detach().cpu())
        if not teachers or not sum(len(value) for value in teachers):
            raise ValueError("calibration input iterable has no selected prediction positions")
        return {layer: torch.cat(values) for layer, values in activations.items()}, torch.cat(teachers), references

    train_values, train_teacher, references = capture(train_inputs)
    validation_values, validation_teacher, _ = capture(validation_inputs)
    directory = Path(directory)
    method.training_history = {}
    for layer in method.layers:
        hidden = train_values[layer]
        kernel = method.kernels.get(layer)
        if kernel is None:
            device, dtype = references[layer]
            kernel = method._new_kernel(layer, hidden.shape[-1], device, dtype,
                hidden.shape[-2] if method.name == "attention" else None,
                initial_unembedding=initial_unembedding, initial_bias=initial_bias)
        generator = torch.Generator().manual_seed(training_options.get("seed", 0))
        train_batches = DataLoader(_Readouts(hidden, train_teacher), batch_size=batch_size,
                                   shuffle=True, generator=generator)
        validation_batches = DataLoader(_Readouts(validation_values[layer], validation_teacher),
                                        batch_size=batch_size)
        options = dict(training_options)
        options["provenance"] = {**dict(options.get("provenance") or {}),
            "batch_size": batch_size, "tokens": str(method.tokens),
            "train_positions": len(train_teacher), "validation_positions": len(validation_teacher)}
        method.training_history[layer] = train_distribution_lens(kernel, train_batches,
            validation_batches, directory / f"layer_{layer}", **options)
        method.kernels[layer] = kernel
    method.save(directory)
    return method
