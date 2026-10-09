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

Project URL: https://gexinh.github.io/vlm-probing/

The repository README includes a Project Page badge and text link to this URL.
The workflow at .github/workflows/deploy-project-page.yml publishes this folder
at the site root, so styles, JavaScript, SVGs, and JSON use the same relative paths.

### One-time setup (repository administrator)

1. Open repository Settings → Pages.
2. Under Build and deployment, choose **GitHub Actions** as the Source.
3. Open Actions → Deploy project page → Run workflow on **main**.
4. Wait for both build and deploy to succeed, then use **Visit site** in Pages settings.
5. In the repository About panel, set Website to https://gexinh.github.io/vlm-probing/.

Subsequent changes to this folder on main publish automatically.
The workflow uploads only index.html, styles.css, app.js, and assets;
Python source, notebooks, model weights, and the rest of the private repository
are not part of the website artifact.

Private repositories require an eligible GitHub plan for Pages. If Pages settings
offer only an upgrade or public-repository option, keep this research repository
private and use a separate public page repository containing the site assets,
then update the README links to that repository's Pages URL.

A published page exposes its HTML, JavaScript, diagrams, and displayed experiment
numbers. Repository, documentation, notebooks, and technical report links still
require repository access while the research repository is private.

### Verification

Visit the Project Page badge from the repository README. Confirm that the
framework image loads, route selection and layer changes update the archived
experiment, and Copy code works. If deployment succeeds but the site returns 404,
allow up to ten minutes for publication and check the Pages URL and workflow
deployment environment.

## Provenance and scope

Initial source: gexinh/vlm-probing main at a60f17a890dd1763575fc28ad51e13c80a67ad00, read on 2026-10-08. Content comes from README.md, docs/README.md, docs/models/README.md, demos/README.md, and pyproject.toml.

The logo and protocol diagram are existing project assets. The experiment was extracted without changing any numerical values from demos/snapshots/attention_knockout/07302654.json. Only presentation-relevant fields are included; machine-local paths and unnecessary metadata are omitted.

The plotted metric is P(first answer token “Y”, ID 612), not probability of the full answer “Yellow”. Intervention windows contain up to nine consecutive layers, clipped at the boundaries, with all heads blocked. The browser replays recorded numbers and does not run a model or new interventions.

Eight VLM adapter families does not mean equal method support or pretrained validation for all eight. The technical report exercises five pretrained VLMs; some families have architecture-level checks. These demos do not establish dataset-level performance.

No generated images, third-party sample photographs, analytics, authentication code, model weights, or external runtime services are included in the page.