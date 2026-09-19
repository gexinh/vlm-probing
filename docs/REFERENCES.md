# Paper and Source Index

[Home](../README.md) / [Documentation](README.md)

References below identify the algorithms and related VLM applications. Method
pages describe exactly what is implemented. The library supplies reusable
computations; it does not claim to reproduce every paper's training recipe,
evaluation pipeline, or reported benchmark results.

## Logit Lens

[Interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens),
nostalgebraist, 2020. [Method guide](methods/logit_lens.md).

Layer-wise readout using the final normalization and unembedding.

## Tuned Lens

[Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112).
[Original code](https://github.com/AlignmentResearch/tuned-lens) ·
[Method guide](methods/tuned_lens.md).

Implemented: affine translation with output-distribution KL calibration.

## Attention Lens

[Attention Lens: A Tool for Mechanistically Interpreting the Attention Head Information Retrieval Mechanism](https://arxiv.org/abs/2310.16270).
[Original code](https://github.com/msakarvadia/AttentionLens) ·
[Method guide](methods/attention_lens.md).

Implemented: learned head-specific vocabulary decoders calibrated jointly through
summed logits. Fixed readout attribution is documented separately below.

## Jacobian Lens

[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html).
[Original code](https://github.com/anthropics/jacobian-lens) ·
[Method guide](methods/jacobian_lens.md).

Implemented: averaged downstream Jacobian transport and model readout, with
explicit source/target-position selection.

## EmbedLens

[What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models](https://arxiv.org/abs/2603.00510).
[Original code](https://github.com/EIT-NLP/EmbedLens) ·
[Method guide](methods/embed_lens.md).

Implemented: input-embedding cosine neighbors and optional caller-defined token
groups. Full clustering, pruning, and sink/dead/alive evaluation are outside this release.

## Patchscopes

[Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models](https://arxiv.org/abs/2401.06102).
[Original code](https://github.com/PAIR-code/interpretability/tree/master/patchscopes) ·
[Method guide](methods/patchscope.md).

Implemented: source-to-target residual patching with optional mapping and
teacher-forced output logits.

## Activation Patching

[Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
provides a causal-tracing application.
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and
[pyvene](https://github.com/stanfordnlp/pyvene) are related intervention libraries.
[Method guide](methods/activation_patching.md).

Implemented: aligned residual replacement with measured output-score effects.

## Attribution Patching

[Attribution Patching: Activation Patching at Industrial Scale](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching),
Neel Nanda.
[Method guide](methods/attribution_patching.md).

Implemented: receiver-side gradient times source-minus-receiver activation.

## EAP-IG

[Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms](https://arxiv.org/abs/2403.17806).
[Original code](https://github.com/hannamw/EAP-IG) ·
[Method guide](methods/eap_ig.md).

The implementation here integrates along a simultaneous **edge-activation**
path with trapezoidal quadrature. The paper/repository's input-interpolation
algorithm and automatic circuit discovery are distinct procedures.

## Attention Knockout

[Cross-modal Information Flow in Multimodal Large Language Models](https://arxiv.org/abs/2411.18620).
[Original code](https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM) ·
[Method guide](methods/attention_knockout.md).

Implemented: mask selected attention paths at an editable pre-softmax site.

## Rollout

[Quantifying Attention Flow in Transformers](https://arxiv.org/abs/2005.00928).
[Original code](https://github.com/samiraabnar/attention_flow) ·
[Method guide](methods/attention_rollout.md).

Implemented: residual-aware, row-normalized self-attention rollout.

## Relevance

[Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers](https://arxiv.org/abs/2103.15679).
[Original code](https://github.com/hila-chefer/Transformer-MM-Explainability) ·
[Method guide](methods/attention_relevance.md).

Implemented: the positive-gradient self-attention recurrence. Full cross-modal
relevance propagation and layerwise relevance propagation are not implemented.

## Visual Attention

[See What You Are Told: Visual Attention Sink in Large Multimodal Models](https://arxiv.org/abs/2503.03321)
and [OPERA](https://arxiv.org/abs/2311.17911) are related attention-analysis applications.
[Profile guide](methods/attention_profile.md).

The library provides entropy, concentration, and grouped probability mass as
generic statistics, without the complete VAR or OPERA decoding algorithms.

## Head Attribution

[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
provides background for reading attention-head contributions.
[Head logit attribution](methods/head_logit_attribution.md) ·
[Attention reweighting](methods/attention_reweight.md).

These are reusable fixed-readout and probability-editing primitives; there is no
fitted head decoder in head logit attribution.

## Steering

[The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models Via Visual Information Steering](https://proceedings.mlr.press/v267/li25ca.html).
[Original code](https://github.com/LzVv123456/VISTA) ·
[Method guide](methods/steering.md).

Implemented: add a supplied residual direction with optional norm preservation.
VISTA's direction construction and full evaluation are separate.

## Generic Primitives

[Ablation](methods/ablation.md) implements zero, mean, and supplied-reference
replacement. [Attention temperature](methods/attention_temperature.md) rescales
masked attention logits. These operations do not have a unique originating paper.

A related temperature-intervention application is
[Compose and Fuse: Revisiting the Foundational Bottlenecks in Multimodal Reasoning](https://arxiv.org/abs/2509.23744)
([code](https://github.com/DELTA-DoubleWise/OmniReason)); its probes, data, and
reasoning pipeline are outside the library's implementation scope.
