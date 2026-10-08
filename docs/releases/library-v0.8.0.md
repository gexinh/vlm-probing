# VLM Probing 0.8.0: reusable visualization and portable demos

## Goal

Publish the current probing library with accurate method documentation,
reusable visualizations, portable worked examples, and its technical report.
The release should let a reader inspect a real result immediately and rerun
the corresponding analysis with an explicitly selected model and input.

## Changes from the previous version

The previous GitHub snapshot was `9a09f50` and predates the local library's
path patching, expanded attention methods, calibrated-lens training, and
transformer-graph attribution work. This release includes those completed
library changes and adds a dedicated visualization package. It replaces the
obsolete ablation entry with IOI-style controlled path patching.

- Update method, model, API, and training guides against the current code.
  Separate paper-based methods, generic utilities, and partial method
  transfers. Correct paper mappings and add dataset references.
- Add reusable visualization functions for lens readouts, image attribution,
  causal effects, input displays, curves, and measured paths. Plotting uses
  explicit values and coordinates; it does not depend on a particular model,
  dataset, or report directory.
- Consolidate user-facing notebooks, small saved results, and required sample
  images under `demos/`. Keep saved-result replay separate from optional live
  runs that need model weights or calibrated-lens artifacts.
- Link the technical report from the README and publish its final PDF as a
  GitHub Release attachment. Keep the report build tree, papers, research,
  slides, model weights, and bulk experiment outputs outside Git.

The release retains the existing family classes and model-bound `.run(inputs)`
interface. Generic CPU examples remain in `examples/`; report and slide
generation scripts remain local. Vendored third-party source retains its
upstream license and provenance.

## Expected experimental results

The new presentation and plotting entry points should reproduce the saved
values without changing the probing algorithms. Real model outputs retain
their original labels and scores. Readouts, gradient estimates, and measured
interventions remain distinct; a changed fixed-answer score is not presented
as a verified change in free generation.

No new pretrained-model experiment, fitting run, or dataset-level performance
claim is introduced by this publication. Reruns require the documented model
and data preparation, and fitted methods require an author checkpoint or
separate calibration and held-out evaluation inputs.

## Technical report

[VLM Probing technical report](https://github.com/gexinh/vlm-probing/releases/latest/download/vlm-probing-technical-report.pdf)

The repository and release remain private. The report attachment contains the
finished document, not the generated report source/output directory.

## Validation

Checks were run from a separate checkout containing only the staged publication
files, so local report and experiment scripts could not satisfy dependencies.

- 216 library and visualization tests passed.
- All nine default offline notebooks executed successfully; the committed
  notebooks contain 39 PNG outputs and no error outputs.
- 29 image and compact-measurement hashes match the demo manifest.
- 378 documentation relative links/anchors were checked, 64 Python blocks
  parsed, and 19 executable method examples passed.
- The 0.8.0 wheel built and installed; imports, lazy plotting dependencies, and
  rendering passed from the installed package.
- The staged file list excludes generated paper, report, research, and slides
  directories. Whitespace checks passed and third-party license notices remain.

The report PDF has 29 pages. Its SHA-256 is
`c4cfdaff24ebe687670e2e4c33c865bd28dc40ad2e77a247c1596bcacc4b14a7`.
Private repository and Release attachment checks are performed after upload.
