# Documentation

[Home](../README.md)

Start with a model guide, prepare its native inputs, and select a method from the
three families. The same result format feeds the plotting functions and saved
demo workflows.

- [Model guides and capability matrix](models/README.md): loading, native inputs, supported sites, and validation scope.
- [Method guides](methods/README.md): computation, examples, parameters, outputs, and paper mappings.
- [Common API](API.md): token selectors, complete-answer metrics, alignment, fitting, and persistence.
- [Visualization](VISUALIZATION.md): reusable lens heatmaps, image overlays, causal grids, curves, and graphs.
- [Lens calibration](TRAINING.md): training and validation splits, author checkpoints, persistent optimization, and resumable Jacobian estimation.
- [Package architecture](ARCHITECTURE.md): family base classes, adapters, execution, and results.
- [Custom adapters](ADAPTERS.md): connect another architecture through an explicit contract.
- [Papers and datasets](REFERENCES.md): original algorithms, author code, and data sources.
- [Demos](../demos/README.md): offline replay of measured examples, input images, and opt-in model reruns.
- [Technical report](https://github.com/gexinh/vlm-probing/releases/latest/download/vlm-probing-technical-report.pdf): shared-case comparisons across five pretrained VLMs, plus language and vision controls.

## Choose a demo

| Question | Notebook |
| --- | --- |
| How do residual and head readouts differ? | [Text lens comparison](../demos/lens_comparison_demo.ipynb) |
| What do VLM layers and image embeddings decode to? | [VLM lens comparison](../demos/vlm_lens_demo.ipynb) |
| How do spatial attribution rules differ on one image? | [ViT attention comparison](../demos/attention_comparison_demo.ipynb) · [VLM attention comparison](../demos/vlm_attention_comparison_demo.ipynb) |
| Which residual positions support an answer? | [Activation and attribution patching](../demos/activation_patching_demo.ipynb) |
| Which attention route supports an answer? | [Attention Knockout](../demos/attention_knockout_demo.ipynb) |
| Which controlled head path changes a selected score? | [Path patching](../demos/path_patching_demo.ipynb) |
| Do selected edges recover an IOI answer margin? | [EAP / EAP-IG](../demos/eap_ig_demo.ipynb) |
| How does residual steering change a caption? | [Steering / VSV](../demos/steering_vsv_demo.ipynb) |

Notebook outputs come from real runs on the cited checkpoints and samples.
Replay uses bundled compact measurements; rerunning requires the model and,
for fitted lenses, a compatible decoder artifact or independent calibration.
