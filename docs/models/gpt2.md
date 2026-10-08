# GPT-2 (IOI reference architecture)

[Home](../../README.md) / [Models](README.md)

The native `GPT2LMHeadModel` adapter supports the text decoder used by Wang et al.
in the IOI circuit study. The example below uses random weights to demonstrate
the API without downloads; reproducing the paper's behavioral results requires
the pretrained GPT-2 small checkpoint and the original IOI datasets.

```python
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from vlm_probing import Prober, TokenMargin

torch.manual_seed(4)
config = GPT2Config(vocab_size=40, n_embd=16, n_head=2, n_layer=2,
                    n_positions=16, attn_pdrop=0.0, resid_pdrop=0.0,
                    embd_pdrop=0.0)
config._attn_implementation = "eager"
model = GPT2LMHeadModel(config).eval()
probe = Prober(model)
base = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
donor = {"input_ids": torch.tensor([[1, 5, 3, 4]])}

lens = probe.lens.logit(layers=[0, 1]).run(base)
path = probe.causal.path(
    senders=[(0, 0)],
    receivers=[(1, 1, "q")],
    sender_tokens="last_prompt",
    receiver_tokens="last_prompt",
).run(base, donor=donor, metric=TokenMargin(4, 5), alignment="position")
print(path.tensors["effect"])
```

Replace the random configuration with
`GPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="eager")`
and tokenize aligned base/donor prompts for pretrained experiments. Check that
the selected token positions refer to the intended IOI roles; positional
alignment does not infer names, subjects, or indirect objects.

## Adapter behavior

The adapter captures each attention head's message at the input to `c_proj`.
Freezing that message freezes its contribution after the fixed output matrix;
the shared projection bias remains unchanged. Q/K/V receivers are separate
slices of the packed `c_attn` output, so replacing Q leaves K and V intact. Final
residual receivers are measured before `transformer.ln_f`.

Logit lenses use the model's final LayerNorm and language head. The adapter does
not expose fixed-scale head logit attribution for GPT-2's affine LayerNorm.
Attention observation requires eager attention; path patching operates on
projection sites and does not require attention probabilities. Cross-attention
configurations require an explicit adapter.

## Validation

Random GPT-2 configurations were tested with Transformers 5.3.0. Validation
checks the packed Q/K/V shapes, the reconstruction of the attention output from
per-head contributions, and equality between final-residual decoding and the
unmodified full forward pass. Four-stage forwards also cover joint Q/K/V and
final-residual receivers in padded batches, including a zero-effect check with
identical base and donor inputs. These checks validate the implementation; they
are not a reproduction of the pretrained IOI circuit.

See [IOI-style path patching](../methods/path_patching.md) and the
[official GPT-2 implementation](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/gpt2/modeling_gpt2.py).

The [executed EAP-IG notebook](../../demos/eap_ig_demo.ipynb) additionally runs
pretrained GPT-2 small on two author-published IOI pairs. Configure
`probe.causal.eap_ig(graph="transformer", steps=5)` for original input-embedding
integration over independent pre-LayerNorm Q/K/V, MLP, and readout messages.
Its 32,491-edge graph supports actual complement-edge replacement to evaluate
retained circuits. This graph currently covers native GPT-2 self-attention;
cross-attention and reordered/upcast attention are rejected.
