#!/usr/bin/env -S uv run --script
"""Cold-path converter for normalized SONIC motion NPZ clips.

The original SONIC PKL/SMPL corpus must first be converted to normalized NPZ
arrays (joint/body order and coordinate convention are task-specific).  This
command then creates the checksummed versioned store consumed by
``train_sonic_unilab.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unilab.training.sonic_motion import (  # noqa: E402
    convert_sonic_motion,
    materialize_motion_store,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        help="Input normalized NPZ (repeatable)",
    )
    parser.add_argument(
        "--normalize-source",
        action="append",
        help="Raw NPZ/PKL/joblib source to normalize before materialization (repeatable)",
    )
    parser.add_argument("--output", required=True, help="Output motion-store directory")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--source-fps",
        type=float,
        default=None,
        help="FPS of raw sources when they do not carry an fps field",
    )
    parser.add_argument("--joint-order", nargs="+", required=True)
    parser.add_argument("--body-order", nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hardlink", action="store_true", help="Use hardlinks when possible")
    parser.add_argument(
        "--clip-id",
        default=None,
        help="Select one nested clip from a multi-clip PKL source",
    )
    parser.add_argument(
        "--quaternion-order",
        choices=("wxyz", "xyzw"),
        default="wxyz",
        help="Quaternion order for raw body/root fields (root_rot is always treated as xyzw)",
    )
    parser.add_argument(
        "--fk-model",
        default=None,
        help="MuJoCo XML used for cold-path FK when raw sources have no body fields",
    )
    parser.add_argument(
        "--smpl-y-up",
        action="store_true",
        help="Convert SMPL root axis-angle from the upstream Y-up convention to Z-up",
    )
    parser.add_argument(
        "--no-derive-velocities",
        action="store_true",
        help="Keep supplied velocity fields instead of finite-difference derivation",
    )
    args = parser.parse_args()
    if bool(args.source) == bool(args.normalize_source):
        parser.error("provide exactly one of --source or --normalize-source")

    sources = args.source
    if args.normalize_source:
        # Keep conversion artifacts in a temporary cold-path directory, then
        # let the existing materializer produce the checksummed immutable store
        # before the temporary directory is removed.
        with tempfile.TemporaryDirectory(prefix="unilab-sonic-normalize-") as temporary_dir:
            normalized_sources = []
            for index, raw_source in enumerate(args.normalize_source):
                normalized_path = Path(temporary_dir) / f"clip_{index:06d}.npz"
                convert_sonic_motion(
                    raw_source,
                    normalized_path,
                    source_fps=args.source_fps,
                    target_fps=args.fps,
                    joint_order=args.joint_order,
                    body_order=args.body_order,
                    clip_id=args.clip_id,
                    quaternion_order=args.quaternion_order,
                    fk_model_path=args.fk_model,
                    derive_velocities=not args.no_derive_velocities,
                    smpl_y_up=args.smpl_y_up,
                )
                normalized_sources.append(normalized_path)
            report = materialize_motion_store(
                normalized_sources,
                args.output,
                fps=args.fps,
                joint_order=args.joint_order,
                body_order=args.body_order,
                overwrite=args.overwrite,
                copy_mode="hardlink" if args.hardlink else "copy",
            )
    else:
        report = materialize_motion_store(
            sources,
            args.output,
            fps=args.fps,
            joint_order=args.joint_order,
            body_order=args.body_order,
            overwrite=args.overwrite,
            copy_mode="hardlink" if args.hardlink else "copy",
        )
    print(
        json.dumps(
            {
                "manifest_path": str(report.manifest_path),
                "clip_count": report.clip_count,
                "total_frames": report.total_frames,
                "total_bytes": report.total_bytes,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
