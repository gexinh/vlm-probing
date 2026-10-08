# Pixtral

[Home](../../README.md) / [Model support](README.md)

Pixtral-12B uses a spatial patch encoder and a Mistral decoder through the native
Transformers `LlavaForConditionalGeneration` class. The automatic adapter supports
residual lenses and interventions, projected heads, IOI-style path patching, and
consumed eager attention scores/probabilities. All seven attention attribution
methods in the technical report were executed on the pretrained checkpoint.
See the [official Transformers guide](https://huggingface.co/docs/transformers/model_doc/pixtral).

```python
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from vlm_probing import Prober, TokenMargin

checkpoint = "/path/to/downloaded/Pixtral-12B-2409"
processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
# Transformers 5.3.x: apply the native Mistral regex correction after loading.
tokenizer = processor.tokenizer
tokenizer._patch_mistral_regex(
    tokenizer, checkpoint, init_kwargs=tokenizer.init_kwargs,
    fix_mistral_regex=True, is_local=True, local_files_only=True,
)
model = LlavaForConditionalGeneration.from_pretrained(
    checkpoint, dtype=torch.bfloat16, attn_implementation="eager"
).to("cuda").eval()
probe = Prober(model, processor)

messages = [{"role": "user", "content": [
    {"type": "image"}, {"type": "text", "text": "Is the shirt black or yellow?"}
]}]
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = probe.prepare(
    prompt=prompt, image=Image.open("example.jpg").convert("RGB"),
    device="cuda", size={"longest_edge": 256}
)
inputs.kwargs["pixel_values"] = inputs.kwargs["pixel_values"].to(torch.bfloat16)
target = processor.tokenizer.encode("Yellow", add_special_tokens=False)
contrast = processor.tokenizer.encode("Black", add_special_tokens=False)
assert len(target) == len(contrast) == 1
metric = TokenMargin(target[0], contrast[0])
result = probe.attention.attribution(
    layers=[39], queries="last_prompt", keys="visual", steps=20
).run(inputs, metric=metric)
```

The snippet applies Transformers 5.3's native Mistral tokenizer correction
after loading: that version's constructor passes the correction flag twice.
Checkpoint files remain unchanged. The report uses a square-padded input and
256-pixel longest edge, giving a verified 16×16 grid. Image-row separators are
text positions, not patch positions. For arbitrary aspect ratios, derive the
grid from processor `image_sizes`; never infer a square grid from token count.
Use `SequenceLogProb` with aligned answer masks for multi-token answers.
