# Demonstrations

These notebooks show what an input is, where a probe acts, and what its readout
or intervention changes. Each uses one or two real examples. The default mode
loads compact **actual model measurements**, redraws them through
`vlm_probing.visualization`, and needs no downloaded model or GPU. The notebooks
retain their executed outputs for GitHub viewing.

Install the notebook environment from the repository root:

```bash
pip install -e '.[notebooks,visualization]'
jupyter lab demos
```

| Notebook | Input and model | Demonstration |
| --- | --- | --- |
| [Text lenses](lens_comparison_demo.ipynb) | Two author IOI prompts; GPT-2 small | Logit, Tuned, all-layer Jacobian, and fitted Attention Lens readouts on the same inputs. |
| [VLM lenses](vlm_lens_demo.ipynb) | Two GQA shirt questions; LLaVA-1.5-7B | Logit/Tuned readouts, visual-token lexical neighbors, and configured Patchscope transfer. |
| [ViT attention](attention_comparison_demo.ipynb) | Author cat/dog and elephant/zebra examples; ViT-B/16 | Eight attention maps and measured patch-deletion curves. |
| [VLM attention](vlm_attention_comparison_demo.ipynb) | The same GQA examples; LLaVA-1.5-7B | Eight decoder attention analyses with a fixed complete-answer score. |
| [Activation and attribution patching](activation_patching_demo.ipynb) | Yellow-shirt GQA input; SmolVLM-500M and LLaVA-1.5-7B | Residual/token sensitivity, measured replacement versus local gradient estimate. |
| [Attention Knockout](attention_knockout_demo.ipynb) | Two boxed GQA examples; LLaVA-1.5-7B | Nine-layer route blocking and target-versus-other image-region scans. |
| [Path Patching](path_patching_demo.ipynb) | Two published VRUBench paper prompts; Qwen2.5-VL-3B/7B | Genuine IOI-style head-output freezing and receiver-only replay. |
| [Residual Steering / VSV](steering_vsv_demo.ipynb) | COCO2014 images 310196 and 210789; LLaVA-1.5-7B | Changed captions, fixed-prefix token ranks, and steering-strength responses. |
| [EAP / EAP-IG](eap_ig_demo.ipynb) | Two author IOI pairs; GPT-2 small | Explicit edge scores and actual full-complement circuit recovery. |

The model/data/metric are stated in each notebook. Snapshots preserve native
vocabulary IDs, values, layer coordinates, and source hashes. Replay changes
plotting only; it does not run new interventions or train probes. These small
examples do not support dataset-level performance claims.

## Run a model again

Install `pip install -e '.[transformers,notebooks,visualization]'`, then set
`RERUN = True` in a notebook. Configure a model and a cache if desired:

```bash
export HF_HOME="$HOME/.cache/vlm-probing/huggingface"
export VLM_PROBING_DEVICE="cuda:0"
export VLM_PROBING_MODEL="llava-hf/llava-1.5-7b-hf"
```

Leave `VLM_PROBING_MODEL` unset to use the notebook's default model. The ViT demo
instead accepts `VLM_PROBING_VIT_MODEL`; its default public Hugging Face checkpoint
differs from the archived author's checkpoint. No model weights are bundled.
The optional run cells make actual public `Prober` calls. Full-head and joint
window sweeps perform many forwards and are intentionally opt-in.

Model loading, dataset preparation, probe configuration, and plotting are
separate. For a different dataset, prepare native inputs and explicit masks;
then reuse the same probe and plotting APIs. The GQA preparation in
[inputs.py](inputs.py) is specifically the paper-aligned LLaVA square-pad recipe;
other processors should supply their own input coordinates. A compatible model
adapter is still required: inspect `probe.describe()` before a costly run.

Tuned, Attention, and Jacobian lenses require matching fitted artifacts. Set
`VLM_PROBING_LENS_CONFIG` to a JSON file following
[lens-config.example.json](lens-config.example.json); directories resolve beside
that config. Use the exact binding metadata recorded when fitting or importing
the artifact. The configuration example contains explanatory placeholders, not
trained weights. Logit Lens, lexical EmbedLens, and configured Patchscopes do
not require fitted decoders. See the [training guide](../docs/TRAINING.md) for
independent calibration/validation workflows; do not fit on the two display
examples.

## Data and provenance

| Source | Included samples and role |
| --- | --- |
| [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html), [Cross-modal Information Flow author subset](https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM/tree/main/datasets) | Question 07302654 / image 2341576 / yellow shirt; question 19225230 / image 2342134 / black shirt. Original images, object boxes, and question metadata are bundled. |
| [COCO2014](https://cocodataset.org/#download), [VISTA POPE records](https://github.com/LzVv123456/VISTA/tree/main/pope_coco) | Images `COCO_val2014_000000310196.jpg` and `COCO_val2014_000000210789.jpg`. Image-specific source URLs and hashes are in `assets/coco_samples.json`. |
| [Transformer Explainability author samples](https://github.com/hila-chefer/Transformer-Explainability/tree/main/samples) | `catdog.png` and `el2.png`, plus their exact processed 224×224 crops. These are illustrative author samples, not a claimed held-out ImageNet split. |
| [Author IOI pairs](https://github.com/hannamw/eap-ig-faithfulness/tree/main/data/ioi) | Published GPT-2 CSV rows 826 and 475, preserving the author clean/corrupted prompts and BOS convention in the EAP-IG experiment. |
| [VRUBench paper examples](https://arxiv.org/abs/2604.15294) | Appendix A two-step and Figure 7(c) four-step examples with attributed prompt text and paired changes. These are not a download of the full official benchmark. Release availability in the provenance file is a historical collection-time check. |
| [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext), BookCorpus | Independent calibration sources; raw corpora are not bundled. GPT-2 Attention Lens used 4,096 BookCorpus contexts; Jacobian used 256 WikiText-2 train contexts. Validation/test each use 512 separate WikiText-2 contexts. |
| COCO2014 train / validation | LLaVA Tuned Lens used 1,024 image-caption training examples and 256 separate validation examples, eight text-prediction positions per image; weights and calibration images are not bundled. |

[assets/manifest.json](assets/manifest.json) records the hashes of bundled
measurements/images and the original archive hashes. Machine-local cache paths
are redacted in portable snapshots; measured numbers are unchanged. Source
HTTP headers are omitted. The ViT snapshots point to image files rather than
embedding duplicate base64 PNGs.

Third-party photos and data retain their original rights; bundling does not
change their terms. See [image source notices](assets/images/SOURCE_NOTICES.md).
Method papers and implemented scope are indexed in
[References](../docs/REFERENCES.md). These notebooks contain result snapshots,
not trained checkpoints, full datasets, original papers, slide decks, or report
build outputs.
