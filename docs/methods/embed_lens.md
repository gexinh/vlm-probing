# EmbedLens

[Home](../../README.md) / [Methods](README.md) / EmbedLens

Find vocabulary embeddings nearest to projected visual tokens.

## Implementation

Capture the language decoder's input after image/text merging. Compare normalized vectors with the input embedding table by cosine similarity, processing vocabulary blocks to bound temporary memory. Optional checkpoint-specific token groups label nearest neighbors.

## Example

Use the [small CPU setup](README.md#example-setup) first. For a VLM, use its
[model guide](../models/README.md) and choose layers reported by `describe()`.

```python
result = probe.lens.embed(tokens="visual").run(inputs, top_k=3)
print(result.tensors["token_ids"])
print(result.tensors["similarities"])
print(result.tensors["positions"])
```

## API

```text
probe.lens.embed(*, tokens="visual", token_groups=None)
method.run(inputs, *, top_k=10, vocabulary_chunk_size=8192)
```

| Parameter | Meaning |
| --- | --- |
| `tokens` | Visual positions by default; also accepts any common token selector. |
| `token_groups` | Optional `{name: [vocabulary IDs]}` with nonempty, disjoint groups. |
| `top_k` | Number of vocabulary neighbors. |
| `vocabulary_chunk_size` | Number of embedding rows per comparison block. |

Returns a `ProbeResult`: `token_ids[1,N,K]`, `similarities[1,N,K]`, `activation_norms[1,N]`, positions, and optional group labels. Unmatched groups are `-1`.

## Support and scope

Requires an embedding-space capture and the native input-embedding table.
Supported VLMs provide projected visual tokens; language models can use
explicit text/all-token selectors. This implements semantic nearest-neighbor readout. Sink/dead/alive classification needs checkpoint-specific evidence; no universal token IDs or full paper pruning/clustering pipeline are assumed.

[Paper / source reference](../REFERENCES.md#lens-methods) ·
[Tensor implementation](../../src/vlm_probing/lenses/embed.py) ·
[Shared result conventions](README.md#shared-conventions)
