"""Import an author GPT-2 Tuned Lens with explicit checkpoint validation."""
import hashlib
import json
from pathlib import Path

import torch


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_author_tuned_lens(probe, directory, *, base_weight_file,
                            expected_weight_sha256, tokenizer_id,
                            output_directory=None, tokens="text"):
    """Return a model-bound fitted lens and import provenance.

    The author's ``config.json``/``params.pt`` use HF hidden-state indices:
    embeddings=0, after block i=i+1. Our post-block site i receives translator
    i+1. The final block receives an identity map because its state already is
    in the final basis. Never interpret translator 0 as our post-block site 0.

    This importer validates GPT-2 architecture, file SHA, and every loaded base
    parameter against the verified safetensors checkpoint before copying lens
    weights. Explicit tokenizer identity is required. It does not silently
    import author weights into another model or architecture.
    """
    from safetensors import safe_open
    from transformers import GPT2Config

    root = Path(directory)
    config = json.loads((root / "config.json").read_text())
    if getattr(probe.model.config, "model_type", None) != "gpt2":
        raise ValueError("this author importer validates the GPT-2 checkpoint contract only")
    if not tokenizer_id or config.get("lens_type") != "linear_tuned_lens":
        raise ValueError("explicit tokenizer identity and a linear Tuned Lens artifact are required")
    base_config_path = Path(base_weight_file).parent / "config.json"
    base_config = GPT2Config.from_dict(json.loads(base_config_path.read_text()))
    forward_fields = ("n_embd", "n_layer", "n_head", "n_positions", "n_inner",
        "activation_function", "layer_norm_epsilon", "scale_attn_weights",
        "scale_attn_by_inverse_layer_idx", "reorder_and_upcast_attn", "add_cross_attention")
    for field in forward_fields:
        if getattr(probe.model.config, field) != getattr(base_config, field):
            raise ValueError(f"loaded GPT-2 forward configuration differs at {field}")
    actual_sha = _sha256(base_weight_file)
    if actual_sha != expected_weight_sha256:
        raise ValueError("base checkpoint SHA does not match the verified author binding")
    layers = sorted(probe.spec.residuals)
    if layers != list(range(config["num_hidden_layers"])) or probe.model.config.n_embd != config["d_model"]:
        raise ValueError("author lens dimensions/layer count do not match the loaded model")
    checked = 0
    with safe_open(base_weight_file, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        for name, parameter in probe.model.transformer.named_parameters():
            key = name if name in keys else "transformer." + name
            if key not in keys or not torch.equal(parameter.detach().cpu(), checkpoint.get_tensor(key).to(parameter.dtype)):
                raise ValueError(f"loaded base parameter {name} differs from the verified checkpoint")
            checked += 1
        embedding_key = "wte.weight" if "wte.weight" in keys else "transformer.wte.weight"
        if not torch.equal(probe.model.lm_head.weight.detach().cpu(),
                           checkpoint.get_tensor(embedding_key).to(probe.model.lm_head.weight.dtype)):
            raise ValueError("GPT-2 output head differs from the verified tied embedding")
    artifact_sha = _sha256(root / "params.pt")
    binding = {"model_id": f"{config['base_model_name_or_path']}@{config['base_model_revision']}",
        "tokenizer_id": tokenizer_id, "readout_id": f"native-final-norm-head:{actual_sha}",
        "calibration_id": f"author-tuned-lens:{artifact_sha}"}
    method = probe.lens.tuned(layers=layers, tokens=tokens, binding=binding)
    state = torch.load(root / "params.pt", map_location="cpu", weights_only=True)
    expected_keys = {f"{i}.{kind}" for i in range(len(layers)) for kind in ["weight", "bias"]}
    if set(state) != expected_keys:
        raise ValueError("author artifact translator keys do not match its declared layers")
    mapping = {}
    embedding = probe.spec.input_embeddings
    for layer in layers:
        kernel = method._new_kernel(layer, config["d_model"], embedding.device, embedding.dtype)
        index = layer + 1
        if index < len(layers):
            kernel.translator.load_state_dict({kind: state[f"{index}.{kind}"] for kind in ["weight", "bias"]})
            kernel.is_fitted = True
            mapping[str(layer)] = index
        else:
            mapping[str(layer)] = "identity_final_readout"
        method.kernels[layer] = kernel
    metadata = {"status": "imported", "author_config": config,
        "author_params_sha256": artifact_sha, "base_weight_sha256": actual_sha,
        "base_config_sha256": _sha256(base_config_path),
        "validated_forward_configuration": {field: getattr(base_config, field) for field in forward_fields},
        "checked_runtime_parameters": checked + 1, "binding": binding,
        "post_block_to_author_index": mapping, "author_embedding_translator": "index 0 not imported into a post-block site",
        "final_layer": "untrained identity; native final readout"}
    if output_directory is not None:
        method.save(output_directory)
        (Path(output_directory) / "import.json").write_text(json.dumps(metadata, indent=2))
    return method, metadata
