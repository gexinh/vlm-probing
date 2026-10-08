# Target-conditioned attention maps

[Home](../../README.md) / [Methods](README.md) / Attention maps

These methods explain a fixed model score with attention probabilities,
gradients, or projected value vectors. They answer which input patches or
tokens contribute to that score under a particular approximation. A heatmap
alone does not establish a causal circuit; use [Attention Knockout](attention_knockout.md)
or [Path Patching](path_patching.md) to test specific connections.

## A common call pattern

Bind the model once, prepare its native inputs, choose a target score, and
call `probe.attention.<method>(...).run(inputs, metric=metric)`. The example
below uses native Hugging Face ViT classification, matching the modality of
the TAM and Beyond Intuition image experiments.

```python
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification
from vlm_probing import ClassScore, Prober

model_id = "google/vit-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_id)
model = ViTForImageClassification.from_pretrained(
    model_id, attn_implementation="eager"
).eval()
inputs = dict(processor(images=Image.open("example.jpg").convert("RGB"),
                        return_tensors="pt"))
probe = Prober(model, processor=processor)
print(probe.describe()["methods"])

# Fix this class throughout integration; do not reselect argmax at each step.
with torch.no_grad():
    class_id = model(**inputs).logits[0].argmax().item()
metric = ClassScore(class_id, kind="logit")
last = model.config.num_hidden_layers - 1

cam = probe.attention.grad_cam(
    layers=[last], queries=[0], keys="visual"
).run(inputs, metric=metric)

attattr = probe.attention.attribution(
    layers=[last], queries=[0], keys="visual", steps=20, quadrature="right"
).run(inputs, metric=metric)

tam = probe.attention.tam(
    queries=[0], keys="visual", steps=20, input_key="pixel_values"
).run(inputs, metric=metric)

ours_h = probe.attention.beyond_intuition(
    variant="head", queries=[0], keys="visual", steps=20,
    input_key="pixel_values"
).run(inputs, metric=metric)

ours_c = probe.attention.beyond_intuition(
    variant="token", queries=[0], keys="visual", steps=20,
    input_key="pixel_values"
).run(inputs, metric=metric)

# ViT token 0 is CLS; visual keys 1: are row-major patches.
patch_scores = tam.tensors["map"][0, 0, 1:]
patch_size = model.config.patch_size
height = inputs["pixel_values"].shape[-2] // patch_size
width = inputs["pixel_values"].shape[-1] // patch_size
heatmap = patch_scores.reshape(height, width)
tam.save("outputs/my_attention_map")
```

The query/key selectors preserve full token axes and set unselected map
entries to zero. For ViT, the map defaults are CLS query and visual patch
keys; for a language model or VLM, they are the last prompt position and all
keys. Explicit selectors make comparisons easier to review. `layers=None`
uses supported attention layers. Propagation requires a consecutive stack
starting at layer 0; `start_layer` skips initial layers within that stack.

Use the [model guides](../models/README.md) for VLM preprocessing. With a
prepared `probe`, `inputs`, and verified candidate token IDs, the same public
methods apply to an autoregressive answer:

```python
from vlm_probing import TokenMargin

metric = TokenMargin(positive=correct_token_id, negative=contrast_token_id)
result = probe.attention.grad_cam(
    layers=[last_decoder_layer], queries="last_prompt", keys="visual"
).run(inputs, metric=metric)
image_scores = result.tensors["cam"][0, 0, prediction_position, visual_positions]
```

For answers containing multiple tokens, use an explicitly aligned
`SequenceLogProb` target instead of assuming a word has one token. VLM token
positions refer to the expanded language sequence, including image tokens.
Spatially reshape a visual map only when the processor and token layout
provide a verified patch grid. Multi-crop visual inputs may need several
grids. These decoder maps explain language-side image tokens; they are
different from a vision encoder's CLS-to-patch attention.

## What each method computes

| Method | Model-bound factory | Principle |
| --- | --- | --- |
| Attention Grad-CAM | `probe.attention.grad_cam(...)` | Average score gradients over selected keys within each head/query, weight attention by those head weights, average heads, then rectify. |
| Hao ATTATTR | `probe.attention.attribution(...)` | Integrate score gradients while replacing all heads of one layer by `alpha * original_attention`; multiply the integral by original attention. |
| TAM | `probe.attention.tam(...)` | Propagate the original final attention backwards through earlier residual attention transitions; multiply by positive integrated final-attention gradients along an input path. |
| Beyond Intuition / Ours-H | `probe.attention.beyond_intuition(variant="head", ...)` | Build forward residual perception maps using head importance derived from `abs(A.T @ gradient)`; apply input-path final-attention feedback. |
| Beyond Intuition / Ours-C | `probe.attention.beyond_intuition(variant="token", ...)` | Build perception maps using projected-value norm / pre-LN input norm as source-token weights; apply the same feedback. |

ATTATTR integrates in **attention probability space** with a fixed input.
Its zero baseline prevents that layer from attending anywhere. Probabilities
are deliberately not renormalized during integration. Each selected layer
gets an independent run, so an attribution sweep across 12 layers with 20
steps requires 240 interventions.

TAM and Beyond Intuition integrate in **model input space**. Their default
path scales processed `pixel_values` from zero to the original tensor while
keeping other inputs and the output target fixed. Zero normalized pixels do
not necessarily correspond to a black raw image. A caller-supplied baseline
can be passed to `.run(inputs, metric=metric, baseline=baseline_pixels)`;
the baseline must match the processed tensor's shape, device, and dtype.
The original papers' input-path feedback is an integrated attention gradient,
without multiplying it by attention as ATTATTR does.

These methods need no probe training or pretrained probe checkpoint. The
model weights remain caller-owned; attribution obtains activation gradients
without fitting parameters. Eager attention is required, and the adapter
must expose consumed probabilities rather than detached diagnostic outputs.

## API and returned tensors

```text
probe.attention.grad_cam(*, layers=None, queries=None, keys=None, normalize=False)
probe.attention.attribution(*, layers=None, queries=None, keys=None,
                            steps=20, quadrature="right")
probe.attention.tam(*, layers=None, queries=None, keys=None,
                    steps=20, quadrature="right", input_key="pixel_values",
                    start_layer=0)
probe.attention.beyond_intuition(*, variant="head", layers=None,
                                queries=None, keys=None, steps=20,
                                quadrature="right", input_key="pixel_values",
                                start_layer=0)
method.run(inputs, *, metric, baseline=None)
```

`baseline` applies to input-path methods only. `normalize` is optional
per-query min-max display normalization for Grad-CAM; raw values are the
default. `start_layer` is zero-based inclusive. For the token approximation,
the adapter must additionally capture block inputs and project **unweighted
value vectors** through the output weights with no output bias. Attention
head outputs after `A @ V` are not a substitute.

| Result | Fields and shapes |
| --- | --- |
| Grad-CAM | `cam[L,B,Q,K]`; `head_weights[L,B,H,Q]` |
| ATTATTR | signed `attribution[L,B,H,Q,K]`; `head_attribution[L,B,H]` is the sum over selected query/key pairs; `token_attribution[L,B,Q,K]` is the sum over heads |
| TAM | `state[B,T,T]`, `feedback[B,T,T]`, `relevance[B,T,T]`; selected `map[B,T,T]` |
| Beyond Intuition | `perception[B,T,T]`, `feedback[B,T,T]`, `relevance[B,T,T]`; selected `map[B,T,T]`; `layer_perception[P,B,T,T]`; head `head_weights[P,B,H]` or token `token_weights[P,B,T]` |
| Shared | `query_mask[B,T]`, `key_mask[B,T]`, `baseline_score[B]`; JSON-safe model, metric, layer and integration metadata |

`L` is the number of individually analyzed layers; `P` is the number of
propagated layers at or after `start_layer`; `B` is batch size; `H` is head
count; `T/Q/K` are complete token axes. ATTATTR's head sum is a convenience
aggregation, not the original paper's maximum-based head-pruning importance
metric. All model-bound results default to detached CPU tensors.

## Original settings and this library's defaults

| Method | Original paper/code | Library behavior |
| --- | --- | --- |
| Grad-CAM | The official ViT baseline uses last-layer CLS-to-patch attention, averages gradients over patch keys, and min-max normalizes the display. | Specify the same query/key selections; `normalize=True` reproduces the display normalization. It is an attention-map adaptation of feature Grad-CAM. |
| ATTATTR | Paper Equation 4: right Riemann endpoints, 20 steps. Official code: left endpoints, default 64 steps. | `quadrature="right", steps=20` follows the paper; `quadrature="left", steps=64` matches the code sampling. |
| TAM | Paper: half-normalized residual transitions, right Riemann integration. Official code: unnormalized residual transitions and uniformly averaged endpoint-inclusive input samples. | Model-bound default follows the paper. The tensor kernel's `residual_normalize=False` matches the code scale; it differs by a global power of two, leaving an independently normalized heatmap unchanged. |
| Beyond Intuition | Official generator: input-path integration includes both endpoints, default 20 samples; effectively one-based `start_layer`. | Library indices are zero-based; `start_layer=k-1` matches official `k`. Head weights are normalized independently per example. A zero-gradient head group gets zero weights instead of dividing by zero. |

`quadrature="endpoints"` uses uniform samples including both `0` and `1`
(at least two samples) for the official TAM/Beyond Intuition sampling rule.
This is an equally weighted sample average, not a trapezoidal quadrature.

The standalone tensor classes accept an already averaged
`integrated_gradients[B,H,Q,K]`. They do not infer or run the integration
path themselves. Grad-CAM uses ordinary gradients from the original forward;
the head-wise Beyond Intuition approximation also needs the full original
attention-gradient stack. The [source index](../REFERENCES.md#attention-methods) maps each rule to its
original paper and author code.

## Modality and scope

Hao's original study uses text/BERT. TAM originally uses image classification
with ViT. Beyond Intuition evaluates BERT, ViT and CLIP; an autoregressive
VLM decoder demonstration is a method transfer. Native ViT, supported eager
Llama/LLaVA attention, and custom adapters can use these kernels when their
contracts are met. Hybrid linear-attention layers cannot be skipped as though
they were ordinary softmax attention. Check `probe.describe()` before choosing
methods or layers.

Full Chefer CVPR 2021 DTD/LRP has a separate original-rule ViT backend:

```python
lrp = probe.attention.dtd_lrp(method="transformer_attribution")
result = lrp.run(inputs["pixel_values"], target=class_id)
patch_heatmap = result.tensors["patch_relevance"]
```

The existing `probe.attention.relevance(...)` implements Chefer's ICCV 2021
generic positive-gradient self-attention recurrence. It is a different
algorithm and must not be labeled as full DTD/LRP. Beyond Intuition is cited to the published TMLR 2023 paper; its repository
also retains an earlier 2022 manuscript date.

## References

Executed comparisons: [original ViT control](../../demos/attention_comparison_demo.ipynb)
and [LLaVA/GQA transfer](../../demos/vlm_attention_comparison_demo.ipynb).
Both retain the actual inputs, fixed targets and method configurations.

- [Grad-CAM paper](https://arxiv.org/abs/1610.02391), ICCV 2017;
  [official attention-map baseline](https://github.com/hila-chefer/Transformer-Explainability/blob/c3e578f76b954e8528afeaaee26de3f07e3fe559/baselines/ViT/ViT_explanation_generator.py).
- [Self-Attention Attribution paper](https://arxiv.org/abs/2004.11207), AAAI 2021;
  [official ATTATTR code](https://github.com/YRdddream/attattr/blob/f5a7396bee476504e22424b8f11a5ad4e7e55a00/examples/generate_attrscore.py).
- [Transition Attention Maps paper](https://openreview.net/forum?id=TT-cf6QSDaQ),
  NeurIPS 2021 XAI4Debugging workshop;
  [official code](https://github.com/XianrenYty/Transition_Attention_Maps/blob/8329327179f6dcf76df29e03ae519cd594595678/baselines/ViT/ViT_explanation_generator.py).
- [Beyond Intuition paper](https://openreview.net/forum?id=rm0zIzlhcX);
  [official generator](https://github.com/jiaminchen-1031/transformerinterp/blob/a76854abfcfd66a4f74109225ecfc40e07b5a3ef/ViT/baselines/ViT/ViT_explanation_generator.py)
  and [projection definition](https://github.com/jiaminchen-1031/transformerinterp/blob/a76854abfcfd66a4f74109225ecfc40e07b5a3ef/ViT/baselines/ViT/ViT_new.py).
- [Chefer DTD/LRP paper](https://arxiv.org/abs/2012.09838), CVPR 2021;
  [official repository](https://github.com/hila-chefer/Transformer-Explainability).
