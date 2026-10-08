# Llama (text-only control)

[Home](../../README.md) / [Models](README.md)

The native `LlamaForCausalLM` adapter remains available for comparing VLM behavior
with a text-only decoder. It is not counted as one of the eight VLM families.
This example needs no weights or network access.

```python
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from vlm_probing import Prober

config = LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=32,
                     num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)
config._attn_implementation = "eager"
model = LlamaForCausalLM(config).eval()
probe = Prober(model)
inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
result = probe.lens.logit(layers=[0, 1]).run(inputs)
profile = probe.attention.profile(groups={"text": "all"}).run(inputs)
print(result.tensors["logits"].shape)
```

Use `tokens="all"` or `"last_prompt"` for residual causal methods, and
`tokens="all"` for EmbedLens; the default visual selection is empty.
The same class can be loaded with `LlamaForCausalLM.from_pretrained(...)`.
Tested on random configurations in Transformers 4.57.6 and 5.3.0, including
padding, grouped-query attention, and frozen-parameter gradient relevance.

Transformers 5.3 eager attention additionally supports real pre-softmax knockout
and temperature interventions through weight-free score taps. Select explicit
text token positions as keys; a text-only model has no visual token selection.
Consumed-probability taps also support reweighting and Hao attention attribution.
Input-path TAM/Beyond Intuition require an explicit floating embedding input;
the default `pixel_values` path does not apply to a text-only model.
