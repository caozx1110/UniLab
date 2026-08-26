#!/usr/bin/env -S uv run --script
"""Convert the official SONIC release TRL checkpoint to UniLab format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unilab.algos.torch.sonic_ppo.checkpoint import (  # noqa: E402
    convert_official_sonic_release_checkpoint_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Official sonic_release/last.pt")
    parser.add_argument("--output", required=True, help="Destination UniLab checkpoint")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--model-only", action="store_true", help="Do not carry optimizer state")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--trust-source",
        action="store_true",
        help="Allow pickle loading for a trusted official checkpoint",
    )
    args = parser.parse_args()
    if not args.trust_source:
        parser.error("--trust-source is required because the official checkpoint uses pickle")
    report = convert_official_sonic_release_checkpoint_file(
        args.source,
        args.output,
        horizon=args.horizon,
        include_optimizer=not args.model_only,
        overwrite=args.overwrite,
        trust_source=args.trust_source,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
