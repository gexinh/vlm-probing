# VLM Probing project page

A responsive English research project page with three method families, a framework diagram, model support, nine demonstration links, and an interactive replay of a real Attention Knockout experiment.

## Preview

Serve docs/project-page with any static HTTP server:

```bash
python3 -m http.server 8765 --directory docs/project-page
```

Open http://127.0.0.1:8765/. Use HTTP rather than opening index.html directly because the experiment loads local JSON.

## Edit

- Page text and resource URLs: docs/project-page/index.html.
- Colors, spacing, and responsive layout: docs/project-page/styles.css.
- Confirmed author names in order and affiliation: docs/project-page/assets/site-config.json. Empty fields display nothing.
- Experiment data: docs/project-page/assets/experiment.json; rendering: docs/project-page/app.js.
- Repository, documentation, notebooks, and report links currently require repository access. Do not mark them as publicly available before they actually are.
- No paper title, venue, author identity, affiliation, benchmark, or citation has been invented.

## GitHub Pages

The page uses relative asset paths and works at a GitHub Pages project URL. For a separate page repository, copy the contents of docs/project-page into the repository root and enable Pages using a branch with the root directory as its source. Keep the research repository private unless the team decides otherwise.

An existing private repository can also host the page if its account/plan supports Pages; administration access is needed to configure it. A public page reveals its HTML, JavaScript, diagrams, and extracted measurement numbers even if the research repository is private. Confirm the public presentation content with the team before enabling public hosting.

## Provenance and scope

Initial source: gexinh/vlm-probing main at a60f17a890dd1763575fc28ad51e13c80a67ad00, read on 2026-10-08. Content comes from README.md, docs/README.md, docs/models/README.md, demos/README.md, and pyproject.toml.

The logo and protocol diagram are existing project assets. The experiment was extracted without changing any numerical values from demos/snapshots/attention_knockout/07302654.json. Only presentation-relevant fields are included; machine-local paths and unnecessary metadata are omitted.

The plotted metric is P(first answer token “Y”, ID 612), not probability of the full answer “Yellow”. Intervention windows contain up to nine consecutive layers, clipped at the boundaries, with all heads blocked. The browser replays recorded numbers and does not run a model or new interventions.

Eight VLM adapter families does not mean equal method support or pretrained validation for all eight. The technical report exercises five pretrained VLMs; some families have architecture-level checks. These demos do not establish dataset-level performance.

No generated images, third-party sample photographs, analytics, authentication code, model weights, or external runtime services are included in the page.