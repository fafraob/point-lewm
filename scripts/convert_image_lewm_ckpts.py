"""One-time key migration for the downloaded image LeWM checkpoints.

The checkpoints under checkpoints/image_lewm_* (downloaded from Hugging Face,
``quentinll/lewm-<env>``) were exported by an older transformers, whose ViTModel named its blocks
``encoder.layer.N.attention.attention.query`` etc. The transformers pinned in
this repo's pixi env builds the modernized layout
(``layers.N.attention.q_proj`` / ``mlp.fc1`` / ``mlp.fc2``), so
``load_pretrained`` fails with missing/unexpected keys. The tensors themselves
are identical ViT-tiny weights -- only the names changed -- and the mapping is
one-to-one, so this script rewrites ``weights.pt`` in the new naming, verifies
a STRICT ``load_state_dict`` against the model instantiated from the folder's
own ``config.json``, and keeps the untouched original as
``weights_hf_legacy.bak`` (not ``.pt``: the folder-checkpoint loader requires
exactly one ``.pt`` per folder).

Idempotent: an already-converted checkpoint (strict load succeeds) is skipped.

    pixi run python scripts/convert_image_lewm_ckpts.py \
        [checkpoints/image_lewm_tworoom ...]      # default: all image_lewm_*
"""

import json
import re
import shutil
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate

REPO = Path(__file__).resolve().parent.parent

# Applied in order; the attention-level output.dense must be rewritten before
# the layer-level output.dense falls through to mlp.fc2.
_RULES = [
    (re.compile(r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.query\."), r"encoder.layers.\1.attention.q_proj."),
    (re.compile(r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.key\."), r"encoder.layers.\1.attention.k_proj."),
    (re.compile(r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.value\."), r"encoder.layers.\1.attention.v_proj."),
    (re.compile(r"^encoder\.encoder\.layer\.(\d+)\.attention\.output\.dense\."), r"encoder.layers.\1.attention.o_proj."),
    (re.compile(r"^encoder\.encoder\.layer\.(\d+)\.intermediate\.dense\."), r"encoder.layers.\1.mlp.fc1."),
    (re.compile(r"^encoder\.encoder\.layer\.(\d+)\.output\.dense\."), r"encoder.layers.\1.mlp.fc2."),
    (re.compile(r"^encoder\.encoder\.layer\.(\d+)\."), r"encoder.layers.\1."),
]


def remap_key(key: str) -> str:
    for pat, repl in _RULES:
        new, n = pat.subn(repl, key)
        if n:
            return new
    return key


def build_model(folder: Path):
    with (folder / "config.json").open() as f:
        return instantiate(json.load(f))


def convert(folder: Path) -> None:
    weights = folder / "weights.pt"
    state = torch.load(weights, map_location="cpu")
    model = build_model(folder)

    try:
        model.load_state_dict(state)  # strict
        print(f"{folder.name}: already in the current layout, nothing to do")
        return
    except RuntimeError:
        pass

    new_state = {remap_key(k): v for k, v in state.items()}
    n_moved = sum(1 for k in state if remap_key(k) != k)
    if len(new_state) != len(state):
        raise RuntimeError(f"{folder.name}: remap collided ({len(state)} -> {len(new_state)} keys)")
    model.load_state_dict(new_state)  # strict; raises on any residue

    backup = folder / "weights_hf_legacy.bak"
    if not backup.exists():
        shutil.copy2(weights, backup)
    tmp = folder / "weights.pt.tmp"
    torch.save(new_state, tmp)
    tmp.replace(weights)
    print(f"{folder.name}: remapped {n_moved}/{len(state)} keys, strict load OK, original kept as {backup.name}")


if __name__ == "__main__":
    targets = [Path(a) for a in sys.argv[1:]] or sorted((REPO / "checkpoints").glob("image_lewm_*"))
    if not targets:
        sys.exit("no checkpoints/image_lewm_* folders found")
    for folder in targets:
        if not (folder / "weights.pt").is_file():
            sys.exit(f"{folder}: no weights.pt")
        convert(folder)
