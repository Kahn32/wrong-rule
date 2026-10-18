from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .analysis import analytic, report
from .experiments import active, sweep

def main(argv=None):
    parser = argparse.ArgumentParser(description="Controlled PyTorch research on regularization and transfer")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["sweep", "active"]:
        command = sub.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--out", required=True)
        command.add_argument("--device", choices=["cpu", "cuda", "mps"])
        command.add_argument("--threads", type=int, default=1)
    command = sub.add_parser("analytic")
    command.add_argument("--out", required=True)
    command = sub.add_parser("report")
    command.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    if args.command == "report":
        print(report(args.run)); return
    if args.command == "analytic":
        print(analytic(args.out)); return
    if args.threads < 1:
        parser.error("--threads must be >=1")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    config = json.loads(args.config.read_text())
    if args.device:
        config["device"] = args.device
    device = config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS requested but unavailable")
    runner = active if args.command == "active" else sweep
    out = runner(config, args.out)
    print(report(str(out)))

if __name__ == "__main__":
    main()
