#!/usr/bin/env -S uv run --script
"""Validate and atomically merge immutable SONIC motion-store shards.

The command waits for ``store_00`` through ``store_07`` by default, validates
each complete source manifest (including every clip hash and array contract),
and publishes one complete store.  Clips are hard-linked when the source and
destination share a filesystem; a copy is used only when linking is not
available.  Hard-linked targets reuse the source checksum after inode identity
is confirmed, avoiding a second read of the complete corpus.  The output is
built in a sibling staging directory and renamed only after its manifest,
relative paths, and target-file existence are preflighted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unilab.training.sonic_motion import (  # noqa: E402
    MANIFEST_SCHEMA,
    MANIFEST_VERSION,
    MotionClip,
    MotionManifest,
    MotionManifestError,
    preflight_motion_manifest,
    resolve_manifest_clip_path,
    sha256_file,
)


@dataclass(frozen=True)
class VerifiedShard:
    """One source manifest already checked through the public preflight."""

    name: str
    manifest_path: Path
    manifest: MotionManifest


def _manifest_paths(shard_root: Path, shard_count: int) -> tuple[tuple[str, Path], ...]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    return tuple(
        (f"store_{index:02d}", shard_root / f"store_{index:02d}" / "manifest.json")
        for index in range(shard_count)
    )


def _wait_for_manifests(
    manifest_paths: tuple[tuple[str, Path], ...],
    *,
    poll_seconds: float,
    timeout_seconds: float | None,
) -> None:
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    started = time.monotonic()
    while True:
        missing = [name for name, path in manifest_paths if not path.is_file()]
        if not missing:
            return
        if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f"timed out waiting for manifests: {', '.join(missing)}")
        print(json.dumps({"status": "waiting", "missing_manifests": missing}), flush=True)
        time.sleep(poll_seconds)


def _verify_clip_fps(clip: MotionClip, expected_fps: int) -> None:
    if clip.fps != expected_fps:
        raise MotionManifestError(
            f"clip {clip.id!r} manifest fps={clip.fps} does not match expected {expected_fps}"
        )


def _verify_shards(
    manifest_paths: tuple[tuple[str, Path], ...], *, expected_fps: int
) -> tuple[VerifiedShard, ...]:
    verified: list[VerifiedShard] = []
    expected_fields: tuple[dict[str, object], ...] | None = None
    expected_joint_order: tuple[str, ...] | None = None
    expected_body_order: tuple[str, ...] | None = None
    seen_ids: set[str] = set()

    for name, manifest_path in manifest_paths:
        manifest = preflight_motion_manifest(
            manifest_path,
            verify_checksums=True,
            verify_shapes=True,
        )
        if manifest.schema != MANIFEST_SCHEMA or manifest.version != MANIFEST_VERSION:
            raise MotionManifestError(f"{name} uses an unsupported motion manifest contract")
        fields = tuple(field.to_dict() for field in manifest.fields)
        if expected_fields is None:
            expected_fields = fields
            expected_joint_order = manifest.joint_order
            expected_body_order = manifest.body_order
        elif (
            fields != expected_fields
            or manifest.joint_order != expected_joint_order
            or manifest.body_order != expected_body_order
        ):
            raise MotionManifestError(
                f"{name} fields/shapes/dtypes or joint/body order differ from the first shard"
            )
        for clip in manifest.clips:
            _verify_clip_fps(clip, expected_fps)
            if clip.id in seen_ids:
                raise MotionManifestError(f"duplicate clip id across shards: {clip.id!r}")
            seen_ids.add(clip.id)
        verified.append(VerifiedShard(name, manifest_path, manifest))
        print(
            json.dumps(
                {
                    "status": "verified_shard",
                    "shard": name,
                    "clip_count": len(manifest.clips),
                    "total_frames": sum(clip.num_frames for clip in manifest.clips),
                    "fps": expected_fps,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return tuple(verified)


def _target_name(index: int, source: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem).strip("._") or "clip"
    suffix = source.suffix.lower()
    return f"{index:06d}_{stem}{suffix}"


def _build_store(
    shards: tuple[VerifiedShard, ...], destination: Path, *, expected_fps: int
) -> dict[str, object]:
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    clips_dir = staging / "clips"
    clips_dir.mkdir()
    clips: list[MotionClip] = []
    hardlinked = 0
    copied = 0
    total_frames = 0
    total_bytes = 0
    try:
        for shard in shards:
            for clip in shard.manifest.clips:
                source = resolve_manifest_clip_path(shard.manifest_path, clip.path)
                target = clips_dir / _target_name(len(clips), source)
                try:
                    target.hardlink_to(source)
                except OSError:
                    shutil.copy2(source, target)
                is_hardlink = (
                    target.stat().st_dev == source.stat().st_dev
                    and target.stat().st_ino == source.stat().st_ino
                )
                if is_hardlink:
                    hardlinked += 1
                    checksum = clip.checksum
                else:
                    copied += 1
                    # A fallback copy has distinct storage, so prove it still
                    # has the bytes that passed the source preflight.
                    checksum = sha256_file(target)
                    if checksum != clip.checksum:
                        raise MotionManifestError(
                            f"copied clip {clip.id!r} checksum differs from its verified source"
                        )
                clips.append(
                    MotionClip(
                        id=clip.id,
                        path=str(target.relative_to(staging)),
                        checksum=checksum,
                        fps=expected_fps,
                        num_frames=clip.num_frames,
                        joint_order=clip.joint_order,
                        body_order=clip.body_order,
                    )
                )
                total_frames += clip.num_frames
                total_bytes += target.stat().st_size

        first = shards[0].manifest
        manifest_path = staging / "manifest.json"
        manifest = MotionManifest(
            version=MANIFEST_VERSION,
            schema=MANIFEST_SCHEMA,
            joint_order=first.joint_order,
            body_order=first.body_order,
            fields=first.fields,
            clips=tuple(clips),
            metadata={
                "materializer": "unilab.sonic_motion.shard_merge",
                "source_count": len(clips),
                "shard_count": len(shards),
                "source_shards": [shard.name for shard in shards],
            },
            manifest_path=manifest_path,
        )
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # The complete source preflight already established every hard-linked
        # target's bytes and array contract; fallback copies were re-hashed
        # above.  Keep publication cheap while proving the generated manifest
        # uses only local relative paths and that all targets exist.
        preflight_motion_manifest(
            manifest_path,
            verify_checksums=False,
            verify_shapes=False,
            expected_joint_order=first.joint_order,
            expected_body_order=first.body_order,
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "merged",
        "manifest_path": str(destination / "manifest.json"),
        "clip_count": len(clips),
        "total_frames": total_frames,
        "total_bytes": total_bytes,
        "hardlinked_clip_count": hardlinked,
        "copied_clip_count": copied,
        "fps": expected_fps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-root",
        type=Path,
        default=Path("/data/hdd/home/caozx/ws/datasets/bones-seed/sonic_pair_full_shards"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/hdd/home/caozx/ws/datasets/bones-seed/sonic_motion_store_full"),
    )
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--expected-fps", type=int, default=50)
    parser.add_argument("--wait", action="store_true", help="Wait until every manifest exists")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Optional wait timeout; only used with --wait",
    )
    args = parser.parse_args()
    if args.expected_fps <= 0:
        parser.error("--expected-fps must be positive")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    manifest_paths = _manifest_paths(args.shard_root.expanduser().resolve(), args.shard_count)
    if args.wait:
        _wait_for_manifests(
            manifest_paths,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
    missing = [str(path) for _, path in manifest_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing shard manifests: " + ", ".join(missing))

    shards = _verify_shards(manifest_paths, expected_fps=args.expected_fps)
    result = _build_store(
        shards,
        args.output.expanduser().resolve(),
        expected_fps=args.expected_fps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
