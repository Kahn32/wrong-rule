from __future__ import annotations
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def prepare_output(path: str, command: str, config: dict) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    if (out / "manifest.json").exists():
        raise FileExistsError(f"{out} already contains a run; choose a new output directory")
    source_root = Path(__file__).parent
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_root.glob("*.py"))}
    write_json(out / "manifest.json", {
        "command": command, "config": config, "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(), "source_sha256": hashes,
        "status": "running", "implementation": "independent; not official paper reproduction",
    })
    return out

def complete_output(out: Path):
    path = out / "manifest.json"
    manifest = json.loads(path.read_text()); manifest["status"] = "complete"
    write_json(path, manifest)
