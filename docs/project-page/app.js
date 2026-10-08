'use strict';
const routeSelect = document.getElementById('route-select');
const layerRange = document.getElementById('layer-range');
const chartContainer = document.getElementById('chart-container');
let experiment;

function renderExperiment() {
  const scan = experiment.scan.find(item => item.path === routeSelect.value);
  const center = Number(layerRange.value);
  const index = scan.centers.indexOf(center);
  const probability = scan.probability[index];
  const window = scan.windows[index];
  const percent = value => (value * 100).toFixed(1);
  const delta = (probability - experiment.baseline) * 100;
  const deltaText = (delta >= 0 ? '+' : '−') + Math.abs(delta).toFixed(1);
  document.getElementById('layer-output').textContent = center;
  document.getElementById('window-note').textContent =
    'Blocked layers ' + window[0] + '–' + window[window.length - 1] + ' · all attention heads';
  document.getElementById('knockout-value').innerHTML = percent(probability) + '<span>%</span>';
  const insight = document.getElementById('result-insight');
  insight.replaceChildren();
  const strong = document.createElement('strong');
  strong.textContent = 'Layers ' + window[0] + '–' + window[window.length - 1] + ': ';
  insight.append(strong, 'blocking this route changes P(“Y”) from ' +
    percent(experiment.baseline) + '% to ' + percent(probability) + '% (' +
    deltaText + ' percentage points). Most likely first token after knockout: “' +
    scan.top_predictions[index].replace(/▁/g, ' ').trim() + '”.');
  const chartWidth = Math.max(300, Math.min(620, chartContainer.clientWidth));
  const right = chartWidth - 24;
  const x = value => 52 + value / 31 * (right - 52);
  const y = value => 242 - value * 211;
  const points = scan.probability.map((value, i) => x(scan.centers[i]).toFixed(2) + ',' + y(value).toFixed(2)).join(' ');
  const grid = [0, .25, .5, .75, 1].map(value =>
    '<path d="M52 ' + y(value) + 'H' + right + '" stroke="#e8edf5"/>' +
    '<text x="39" y="' + (y(value) + 4) + '" text-anchor="end" fill="#8994a8" font-size="12">' +
    (value * 100) + '%</text>').join('');
  const ticks = [0, 8, 16, 24, 31].map(value =>
    '<text x="' + x(value) + '" y="268" text-anchor="middle" fill="#8994a8" font-size="12">' + value + '</text>').join('');
  const tooltipX = Math.max(72, Math.min(right - 8, x(center)));
  chartContainer.innerHTML = '<svg viewBox="0 0 ' + chartWidth + ' 302" role="img" aria-labelledby="scan-title scan-description">' +
    '<title id="scan-title">' + scan.label.replace(/→/g, 'to') + ' attention knockout</title>' +
    '<desc id="scan-description">Archived first-token probability across 32 layer-window centers. ' +
    'Selected center ' + center + ': baseline ' + percent(experiment.baseline) +
    ' percent; knockout ' + percent(probability) + ' percent.</desc>' +
    '<g font-family="IBM Plex Mono, monospace">' + grid + ticks +
    '<text x="' + ((right + 52) / 2) + '" y="297" text-anchor="middle" fill="#8994a8" font-size="11">Layer-window center</text></g>' +
    '<path d="M52 ' + y(experiment.baseline) + 'H' + right + '" stroke="#91a0b6" stroke-width="1.5" stroke-dasharray="5 5"/>' +
    '<path d="M' + x(center) + ' 25V242" stroke="#3266e5" stroke-width="1" stroke-dasharray="3 5" opacity=".4"/>' +
    '<polyline points="' + points + '" fill="none" stroke="#3266e5" stroke-width="3" stroke-linejoin="round"/>' +
    '<circle cx="' + x(center) + '" cy="' + y(probability) + '" r="9" fill="#3266e5" opacity=".14"/>' +
    '<circle cx="' + x(center) + '" cy="' + y(probability) + '" r="4.5" fill="#3266e5" stroke="white" stroke-width="2"/>' +
    '<rect x="' + (tooltipX - 27) + '" y="' + (y(probability) - 36) + '" width="54" height="24" rx="4" fill="#3266e5"/>' +
    '<text x="' + tooltipX + '" y="' + (y(probability) - 20) + '" text-anchor="middle" fill="white" font-size="11" font-family="IBM Plex Mono, monospace">' +
    percent(probability) + '%</text></svg>';
}
async function initializeExperiment() {
  try {
    const response = await fetch('assets/experiment.json');
    if (!response.ok) throw new Error('Measurement request failed');
    experiment = await response.json();
    renderExperiment();
    routeSelect.addEventListener('change', renderExperiment);
    layerRange.addEventListener('input', renderExperiment);
    window.addEventListener('resize', renderExperiment);
  } catch (error) {
    routeSelect.disabled = true;
    layerRange.disabled = true;
    document.getElementById('result-insight').textContent =
      'The archived measurement could not be loaded. Reload this page or open the source notebook.';
  }
}
document.getElementById('copy-code').addEventListener('click', async event => {
  const button = event.currentTarget;
  const code = document.getElementById('quickstart-code').textContent;
  try {
    await navigator.clipboard.writeText(code);
    button.textContent = 'Copied';
    window.setTimeout(() => { button.textContent = 'Copy code'; }, 1800);
  } catch (error) {
    const range = document.createRange();
    range.selectNodeContents(document.getElementById('quickstart-code'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    button.textContent = 'Selected — copy manually';
  }
});
async function initializeAuthors() {
  try {
    const response = await fetch('assets/site-config.json');
    if (!response.ok) return;
    const config = await response.json();
    if (!Array.isArray(config.authors) || !config.authors.length) return;
    const line = document.createElement('p');
    line.className = 'authors';
    line.textContent = config.authors.join(' · ');
    if (config.affiliation) {
      const affiliation = document.createElement('small');
      affiliation.textContent = config.affiliation;
      line.append(affiliation);
    }
    document.querySelector('.hero-description').after(line);
  } catch (error) {
    // Author information is optional until confirmed by the team.
  }
}
initializeExperiment();
initializeAuthors();
