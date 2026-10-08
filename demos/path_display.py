"""Dataset-specific prompt cards; measured-path plotting lives in the library."""
from difflib import SequenceMatcher
from html import escape
import re
import math
from vlm_probing.visualization import plot_head_paths, plot_receiver_paths

def sample_title(pair, index=None):
    """Describe the task and ground-truth comparison instead of a source-file ID."""
    source = pair.get("source", {})
    prefix = f"Example {index + 1}: " if index is not None else ""
    if source.get("dataset") == "VRUBench":
        steps = source.get("steps")
        length = f"{steps} rotations" if steps is not None else "Viewpoint rotation"
        description = ("unseen vs known view" if "unknown" in
                       (pair["clean_answer"], pair["donor_answer"])
                       else "revisiting known views")
        title = f"{prefix}{length}, {description}"
    else:
        title = f"{prefix}counterfactual answer comparison"
    return (f"{title}\nExpected answer: {pair['clean_answer']} (clean)"
            f" → {pair['donor_answer']} (donor)")

def _highlight_difference(clean, donor):
    """Escape all prompt text, then mark only the changed character spans."""
    pieces = [[], []]
    tokens = [re.findall(r"\w+|[^\w]+", text) for text in (clean, donor)]
    for operation, clean_start, clean_end, donor_start, donor_end in SequenceMatcher(
            None, *tokens, autojunk=False).get_opcodes():
        for side, text, start, end in ((0, tokens[0], clean_start, clean_end),
                                      (1, tokens[1], donor_start, donor_end)):
            fragment = escape("".join(text[start:end]))
            if operation != "equal" and fragment:
                background = "#ffe4d6" if side == 0 else "#d8edf8"
                fragment = (f'<mark style="background:{background};padding:1px 3px;'
                            f'border-radius:3px;font-weight:700">{fragment}</mark>')
            pieces[side].append(fragment)
    return ["".join(piece) for piece in pieces]

def prompt_cards(pair, *, index=None, clean_rendered=None, donor_rendered=None,
                 token_metadata=None):
    """Return notebook HTML showing both full prompts and expected answers.

    ``clean_rendered`` and ``donor_rendered`` optionally expose the exact native
    model input, including chat delimiters and the assistant generation prefix.
    The scoring candidates are labels; they are never appended to these inputs.
    """
    from IPython.display import HTML

    clean, donor = _highlight_difference(pair["clean_prompt"], pair["donor_prompt"])
    title = escape(sample_title(pair, index).split("\n")[0])
    system = escape(pair.get("source", {}).get("system_prompt", "You are a helpful assistant"))
    columns = []
    for name, prompt, answer, color in (
            ("Clean / base input", clean, pair["clean_answer"], "#b44928"),
            ("Donor / counterfactual input", donor, pair["donor_answer"], "#1f668d")):
        columns.append(
            '<section style="flex:1;min-width:300px;border:1px solid #d9e1e6;'
            'border-radius:9px;background:#fff;padding:16px">'
            f'<div style="color:{color};font-weight:700;margin-bottom:9px">{name}</div>'
            f'<pre style="white-space:pre-wrap;font-size:12px;line-height:1.55;'
            f'margin:0;background:none;border:0;padding:0;color:#243342">{prompt}</pre>'
            '<div style="margin-top:12px;padding-top:10px;border-top:1px solid #e2e8ec">'
            f'<b>Expected answer:</b> <code>{escape(answer)}</code> '
            '(a scoring label, not part of the input)</div></section>')
    details = ""
    if clean_rendered is not None or donor_rendered is not None:
        if clean_rendered is None or donor_rendered is None:
            raise ValueError("provide both rendered model inputs")
        rendered = _highlight_difference(clean_rendered, donor_rendered)
        details = '<details style="margin-top:12px"><summary>Exact native model inputs (chat delimiters included)</summary>'
        for name, text in zip(("Clean", "Donor"), rendered):
            details += (f'<b>{name}</b><pre style="white-space:pre-wrap;font-size:11px;'
                        f'line-height:1.45">{text}</pre>')
        details += '</details>'
    scoring = ("Both interventions and scoring use the last prompt token: the position that predicts "
               "the first answer token after the native assistant prefix. No answer tokens are provided.")
    if token_metadata is not None:
        clean_ids = token_metadata.get("clean_token_ids", [])
        donor_ids = token_metadata.get("donor_token_ids", [])
        scoring += (f" Candidate token IDs: clean {clean_ids}; donor {donor_ids}. "
                    "Only the first token of each candidate is scored.")
    measurements = ""
    if token_metadata is not None:
        baseline, donor_margin = token_metadata.get("clean_margin"), token_metadata.get("donor_margin")
        if baseline is not None and donor_margin is not None:
            baseline, donor_margin = float(baseline), float(donor_margin)
            if not math.isfinite(baseline) or not math.isfinite(donor_margin):
                raise ValueError("displayed baseline/donor margins must be finite")
            measurements = (
                '<div style="margin-top:12px;padding:12px;border-radius:7px;background:#eef3f7;font-size:13px">'
                '<b>Measured model behavior:</b> '
                f'clean margin = {baseline:+.4f}; donor margin = {donor_margin:+.4f}; '
                f'donor − clean = {donor_margin - baseline:+.4f}.<br>'
                'Margin = logit(clean candidate first token) − logit(donor candidate first token). '
                'A positive margin favors the clean candidate’s first token in that context. '
                'Heatmap colors measure movement toward the donor model’s margin; they do not indicate answer accuracy.</div>')
        native_prefix = token_metadata.get("native_assistant_prefix")
        if native_prefix is not None:
            measurements += ('<p><b>Native generation prefix:</b> '
                             f'<code style="white-space:pre-wrap">{escape(str(native_prefix))}</code>'
                             ' ← the final token is the patched prediction position.</p>')
    html = (f'<div style="font-family:system-ui,sans-serif;color:#253544;margin:16px 0">'
            f'<h3 style="margin-bottom:10px">{title}</h3>'
            f'<p><b>System message:</b> <code>{system}</code></p>'
            '<p>Highlighted text is the clean/donor change. The following are the complete user prompts.</p>'
            f'<div style="display:flex;gap:16px;flex-wrap:wrap">{"".join(columns)}</div>'
            f'<p style="font-size:12px;color:#526476">{escape(scoring)}</p>{measurements}{details}</div>')
    return HTML(html)
