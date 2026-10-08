"""Small offline replay helpers; measurement and visualization stay separate."""
from io import BytesIO
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DEMO_ROOT = Path(__file__).resolve().parent
ASSETS = DEMO_ROOT / "assets"
SNAPSHOTS = DEMO_ROOT / "snapshots"


def load_snapshot(name):
    """Read a bundled, completed model measurement; never generate fake data."""
    record = json.loads((SNAPSHOTS / name).read_text(encoding="utf-8"))
    if record.get("status", "complete") != "complete":
        raise ValueError("This snapshot does not contain a completed experiment")
    return record


def load_image(name):
    with Image.open(ASSETS / "images" / name) as opened:
        return opened.convert("RGB")


def show_figure(figure):
    """Embed the generated figure so GitHub shows the real saved notebook output."""
    from IPython.display import Image as NotebookImage, display
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    display(NotebookImage(data=buffer.getvalue()))
    plt.close(figure)


def gqa_input(record):
    """Match the archived LLaVA mean-color square-pad image coordinate system."""
    from demos.inputs import pad_square
    image = load_image(record["sample"]["image_path"])
    mean = [0.48145466, 0.4578275, 0.40821073]
    padded, offset = pad_square(image, mean)
    x, y, width, height = record["sample"]["bbox_xywh"]
    return padded, [x + offset[0], y + offset[1], width, height]


def target_ranks(logits, token_id):
    """Use native vocabulary IDs; ties follow the archived strict-greater rule."""
    import torch
    logits = torch.as_tensor(logits).detach().float().cpu()
    return (1 + (logits > logits[..., token_id, None]).sum(-1)).numpy()
