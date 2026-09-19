# Patchscope

[Home](../../README.md) / [Methods](README.md) / Patchscope

Insert a source representation into a target prompt and inspect its output logits.

## Implementation

Capture a source layer/token residual, optionally map it to a target residual space, replace one target token at a specified layer, and run the target continuation. Each source layer is evaluated independently.

## Example

Use the [small CPU setup](README.md#example-setup) first. For a VLM, use its
[model guide](../models/README.md) and choose layers reported by `describe()`.

```python
scope = probe.lens.patchscope(
    layers=[0], source_position=0, target_layer=0, target_position=2,
)
result = scope.run(inputs, target_inputs=evaluation)
print(result.tensors["logits"].shape)
```

## API

```text
probe.lens.patchscope(*, layers, source_position, target_layer,
                     target_position, target=None, mapping=None)
method.run(inputs, *, target_inputs)
```

| Parameter | Meaning |
| --- | --- |
| `layers / source_position` | Source residual layers and one expanded token position. |
| `target_layer / target_position` | Destination layer and position, valid in every example. |
| `target` | A second Prober; defaults to the source Prober. |
| `mapping` | Callable mapping source to target vectors; required for different model instances. |
| `target_inputs` | Native inputs or ProbeInputs for the target prompt; batch sizes must match. |

Returns a `ProbeResult`: Teacher-forced target `logits[L,B,T_target,V]` plus source/target provenance.

## Support and scope

All seven VLM adapters. The bound method returns full target logits; autoregressive text generation is not implemented. Cross-model vector dimensionality alone does not establish a meaningful mapping.

[Paper / source reference](../REFERENCES.md#patchscopes) ·
[Tensor implementation](../../src/vlm_probing/lenses/patchscope.py) ·
[Shared result conventions](README.md#shared-conventions)
