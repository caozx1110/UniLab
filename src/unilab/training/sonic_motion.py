"""Cold-path validation for versioned SONIC motion manifests.

The manifest is intentionally a metadata contract, not a motion loader.  It
can be parsed during environment materialization and used to validate an NPZ
shard or NPY array before the rollout hot path starts.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

MANIFEST_VERSION = 1
MANIFEST_SCHEMA = "unilab.sonic.motion"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SHAPE_SYMBOLS = {"*", "num_frames", "num_joints", "num_bodies"}


class MotionManifestError(ValueError):
    """Raised when a SONIC motion manifest or clip violates its contract."""


@dataclass(frozen=True)
class MotionMaterializationReport:
    """Summary returned by the cold-path NPZ materializer."""

    manifest_path: Path
    clip_count: int
    total_frames: int
    total_bytes: int


@dataclass(frozen=True)
class MotionConversionReport:
    """Result of converting one raw SONIC clip to a normalized NPZ file.

    The digest is computed *after* writing the normalized artifact.  It is the
    value that should be recorded in :class:`MotionClip` and therefore protects
    the exact bytes consumed by every distributed worker.
    """

    output_path: Path
    checksum: str
    fps: int
    num_frames: int
    fields: tuple[str, ...]


# Source corpora use several names for the same signal.  Keep this table in the
# cold-path converter instead of teaching the rollout loader to probe aliases.
_MOTION_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "joint_pos": (
        "joint_pos",
        "joint_positions",
        "joint_position",
        "dof_pos",
        "dof",
        "qpos",
    ),
    "joint_vel": (
        "joint_vel",
        "joint_velocity",
        "joint_velocities",
        "dof_vel",
        "qvel",
    ),
    "body_pos_w": (
        "body_pos_w",
        "body_pos",
        "body_positions",
        "body_position",
        "global_translation",
        "global_translations",
    ),
    "body_quat_w": (
        "body_quat_w",
        "body_quat",
        "body_quaternions",
        "body_orientation",
        "body_orientations",
        "global_quaternion",
        "global_quaternions",
    ),
    "body_lin_vel_w": (
        "body_lin_vel_w",
        "body_lin_vel",
        "body_linear_velocity",
        "body_linear_velocities",
        "global_velocity",
        "global_velocities",
    ),
    "body_ang_vel_w": (
        "body_ang_vel_w",
        "body_ang_vel",
        "body_angular_velocity",
        "body_angular_velocities",
    ),
    "root_pos": (
        "root_pos",
        "root_position",
        "root_translation",
        "root_trans",
        "root_trans_offset",
        "transl",
        "translation",
    ),
    "root_quat": (
        "root_quat",
        "root_quaternion",
        "root_orientation",
        "root_rot",
        "root_rotation",
    ),
    "root_lin_vel_w": (
        "root_lin_vel_w",
        "root_lin_vel",
        "root_linear_velocity",
    ),
    "root_ang_vel_w": (
        "root_ang_vel_w",
        "root_ang_vel",
        "root_angular_velocity",
    ),
    "smpl_joints": (
        "smpl_joints",
        "smpl_joint_pos",
        "smpl_joint_positions",
        "smpl_global_joints",
    ),
    "smpl_pose": (
        "smpl_pose",
        "smpl_pose_aa",
        "smpl_pose_axis_angle",
    ),
    "smpl_transl": (
        "smpl_transl",
        "smpl_translation",
        "smpl_root_trans",
    ),
    "smpl_root_quat_w": (
        "smpl_root_quat_w",
        "smpl_root_quat",
        "smpl_root_quaternion",
    ),
}

_FPS_ALIASES = ("fps", "frame_rate", "sample_rate", "sampling_rate", "motion_fps")
_PAIR_SOURCE_SUFFIXES = frozenset({".pkl", ".pickle", ".joblib"})


def _scalar_number(value: Any, location: str) -> float:
    """Convert a scalar/zero-dimensional NumPy value to a finite float."""

    import math

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MotionManifestError(f"{location} must be a finite number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise MotionManifestError(f"{location} must be a positive finite number")
    return result


def _load_motion_source(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load one NPZ/PKL/joblib source on the cold path.

    A mapping is copied shallowly so callers can safely reuse their source
    object.  PKL/joblib loading is deliberately confined to this function;
    normalized NPZ files are the only artifacts accepted by the hot-path
    ``MotionLoader``.
    """

    import numpy as np

    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise MotionManifestError(f"motion source does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        try:
            with np.load(path, allow_pickle=False) as archive:
                return {name: archive[name] for name in archive.files}
        except (OSError, ValueError) as exc:
            raise MotionManifestError(f"could not read NPZ motion source {path}: {exc}") from exc
    if suffix == ".npy":
        try:
            return {"joint_pos": np.load(path, allow_pickle=False)}
        except (OSError, ValueError) as exc:
            raise MotionManifestError(f"could not read NPY motion source {path}: {exc}") from exc
    if suffix in {".pkl", ".pickle", ".joblib"}:
        try:
            if suffix == ".joblib":
                import joblib

                value = joblib.load(path)
            else:
                with path.open("rb") as stream:
                    value = pickle.load(stream)
        except Exception as exc:  # noqa: BLE001 - normalize all loader errors
            # Some SONIC releases use joblib-compressed ``.pkl`` files.  Try
            # the release decoder after the stdlib loader rejects the stream.
            if suffix != ".joblib":
                try:
                    import joblib

                    value = joblib.load(path)
                except Exception as joblib_exc:  # noqa: BLE001
                    raise MotionManifestError(
                        f"could not read pickle/joblib motion source {path}: {exc}; "
                        f"joblib fallback: {joblib_exc}"
                    ) from joblib_exc
            else:
                raise MotionManifestError(
                    f"could not read joblib motion source {path}: {exc}"
                ) from exc
        if not isinstance(value, Mapping):
            raise MotionManifestError(
                f"motion source {path} must contain a mapping, got {type(value).__name__}"
            )
        return dict(value)
    raise MotionManifestError(
        f"unsupported motion source format {path.suffix!r}; expected .npz, .npy, .pkl, or .joblib"
    )


def _unwrap_motion_sequence(data: Mapping[str, Any], clip_id: str | None = None) -> dict[str, Any]:
    """Unwrap common ``{clip_name: sequence}`` SONIC PKL layouts."""

    # A direct sequence has at least one known field or a frame-rate marker.
    known: set[str] = set(_FPS_ALIASES)
    known.update(alias for aliases in _MOTION_FIELD_ALIASES.values() for alias in aliases)
    if any(key in data for key in known):
        return dict(data)

    candidates = {str(key): value for key, value in data.items() if isinstance(value, Mapping)}
    if clip_id is not None:
        if clip_id not in candidates:
            raise MotionManifestError(
                f"motion source does not contain clip_id={clip_id!r}; available={sorted(candidates)}"
            )
        return dict(candidates[clip_id])
    if len(candidates) == 1:
        return dict(next(iter(candidates.values())))
    if candidates:
        raise MotionManifestError(
            "motion source contains multiple clips; pass clip_id to select one"
        )
    raise MotionManifestError("motion source does not contain a recognized motion mapping")


def _resolve_motion_field(
    data: Mapping[str, Any],
    canonical: str,
    aliases: Mapping[str, Sequence[str]] | None,
) -> tuple[Any, str | None]:
    custom = tuple(aliases.get(canonical, ())) if aliases is not None else ()
    names = (canonical, *custom, *_MOTION_FIELD_ALIASES.get(canonical, ()))
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        if name in data and data[name] is not None:
            return data[name], name
    return None, None


def _as_array(value: Any, name: str, *, dtype: Any = None) -> Any:
    import numpy as np

    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise MotionManifestError(f"motion field {name!r} is not a numeric array") from exc
    if array.dtype.kind not in "biufc":
        raise MotionManifestError(f"motion field {name!r} must be numeric, got {array.dtype}")
    if array.dtype.kind == "c":
        raise MotionManifestError(f"motion field {name!r} must not be complex")
    return np.asarray(array, dtype=np.float32)


def _validate_frame_axis(array: Any, name: str, frames: int | None) -> int:
    if array.ndim == 0:
        raise MotionManifestError(f"motion field {name!r} must have a frame axis")
    count = int(array.shape[0])
    if count <= 0:
        raise MotionManifestError(f"motion field {name!r} contains no frames")
    if frames is not None and count != frames:
        raise MotionManifestError(f"motion field {name!r} has {count} frames, expected {frames}")
    return count


def _upstream_target_times(num_frames: int, source_fps: float, target_fps: float) -> Any:
    import torch

    duration = (num_frames - 1) * (1 / source_fps)
    return torch.arange(0, duration, 1 / target_fps, dtype=torch.float32)


def _resample_linear(array: Any, source_fps: float, target_fps: float) -> Any:
    import numpy as np
    import torch

    if array.shape[0] <= 1 or abs(source_fps - target_fps) < 1e-9:
        return np.asarray(array, dtype=np.float32).copy()
    values = torch.as_tensor(np.asarray(array, dtype=np.float32), dtype=torch.float32)
    times = _upstream_target_times(array.shape[0], source_fps, target_fps)
    duration = (array.shape[0] - 1) * (1 / source_fps)
    phase = times / duration
    frame_position = phase * (array.shape[0] - 1)
    index_0 = frame_position.floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(array.shape[0] - 1, dtype=torch.long))
    blend = frame_position - index_0.float()
    blend_shape = (blend.shape[0],) + (1,) * (values.ndim - 1)
    blend = blend.reshape(blend_shape)
    result = (1 - blend) * values[index_0] + blend * values[index_1]
    return result.numpy().astype(np.float32, copy=True)


def _normalize_quaternions(array: Any) -> Any:
    import numpy as np

    values = np.asarray(array, dtype=np.float64)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norm <= 1e-8):
        raise MotionManifestError("quaternion field contains a zero-norm quaternion")
    return (values / norm).astype(np.float32)


def _resample_quaternion(array: Any, source_fps: float, target_fps: float) -> Any:
    import numpy as np
    import torch

    values = torch.as_tensor(np.asarray(array, dtype=np.float32), dtype=torch.float32)
    norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    if bool(torch.any(norm <= 1e-8)):
        raise MotionManifestError("quaternion field contains a zero-norm quaternion")
    values = values / norm
    if values.shape[0] <= 1 or abs(source_fps - target_fps) < 1e-9:
        return values.numpy().astype(np.float32, copy=True)
    times = _upstream_target_times(values.shape[0], source_fps, target_fps)
    duration = (values.shape[0] - 1) * (1 / source_fps)
    phase = times / duration
    frame_position = phase * (values.shape[0] - 1)
    index_0 = frame_position.floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(values.shape[0] - 1, dtype=torch.long))
    blend = frame_position - index_0.float()
    flat = values.reshape(values.shape[0], -1, 4)
    qa, qb = flat[index_0], flat[index_1]
    dot = (qa * qb).sum(dim=-1)
    qb = torch.where((dot < 0).unsqueeze(-1), -qb, qb)
    cosine = dot.abs().unsqueeze(-1)
    sine = torch.sqrt(1.0 - cosine * cosine)
    blend = blend[:, None, None]
    result = torch.sin((1 - blend) * torch.acos(cosine)) / sine * qa
    result += torch.sin(blend * torch.acos(cosine)) / sine * qb
    result = torch.where(sine.abs() < 0.001, 0.5 * qa + 0.5 * qb, result)
    result = torch.where(cosine.abs() >= 1, qa, result)
    result = result / torch.linalg.vector_norm(result, dim=-1, keepdim=True)
    return result.numpy().reshape((len(times), *array.shape[1:])).astype(np.float32, copy=True)


def _finite_difference(array: Any, dt: float) -> Any:
    import numpy as np

    values = np.asarray(array, dtype=np.float32)
    if values.shape[0] <= 1:
        return np.zeros_like(values, dtype=np.float32)
    result = np.empty_like(values, dtype=np.float32)
    result[0] = (values[1] - values[0]) / dt
    result[-1] = (values[-1] - values[-2]) / dt
    if values.shape[0] > 2:
        result[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    return result


def _quat_inverse_wxyz(quat: Any) -> Any:
    import numpy as np

    result = np.asarray(quat, dtype=np.float64).copy()
    result[..., 1:] *= -1.0
    return result


def _quat_multiply_wxyz(lhs: Any, rhs: Any) -> Any:
    import numpy as np

    w1, x1, y1, z1 = np.moveaxis(np.asarray(lhs), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(rhs), -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def _quaternion_angular_velocity(quat_wxyz: Any, dt: float) -> Any:
    import numpy as np

    quat = _normalize_quaternions(quat_wxyz).astype(np.float64)
    if quat.shape[0] <= 1:
        return np.zeros((*quat.shape[:-1], 3), dtype=np.float32)
    delta = _quat_multiply_wxyz(quat[1:], _quat_inverse_wxyz(quat[:-1]))
    delta = _normalize_quaternions(delta).astype(np.float64)
    # Choose the shortest representation before extracting angle/axis.
    delta = np.where(delta[..., :1] < 0.0, -delta, delta)
    vector = delta[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(delta[..., :1], 1e-8, None))
    axis = vector / np.where(vector_norm > 1e-8, vector_norm, 1.0)
    velocity = axis * angle / dt
    result = np.empty((*quat.shape[:-1], 3), dtype=np.float32)
    result[0] = velocity[0]
    result[1:] = velocity.astype(np.float32)
    return result


def _axis_angle_to_quaternion_wxyz(axis_angle: Any) -> Any:
    """Convert ``(..., 3)`` axis-angle vectors to WXYZ quaternions.

    This small NumPy implementation mirrors the upstream SONIC conversion
    without importing Torch/IsaacLab during motion materialization.
    """

    import numpy as np

    values = np.asarray(axis_angle, dtype=np.float64)
    if values.shape[-1] != 3:
        raise MotionManifestError("axis-angle values must have a final dimension of 3")
    angle = np.linalg.norm(values, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.divide(
        np.sin(half),
        angle,
        out=np.full_like(angle, 0.5),
        where=angle > 1.0e-8,
    )
    quat = np.concatenate((np.cos(half), values * scale), axis=-1)
    return _normalize_quaternions(quat)


def _derive_smpl_root_quaternion(
    smpl_pose: Any,
    *,
    smpl_y_up: bool,
) -> Any:
    """Reproduce SONIC's SMPL root-frame conversion on the cold path.

    Upstream converts the axis-angle root from Y-up to Z-up (when configured)
    and removes SMPL's fixed ``[0.5, 0.5, 0.5, 0.5]`` rest rotation.  The
    resulting WXYZ sequence is what ``smpl_root_ori_b_multi_future`` consumes.
    """

    import numpy as np

    pose = np.asarray(smpl_pose, dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] < 3:
        raise MotionManifestError("smpl_pose must have shape (T,72) or at least (T,3)")
    root = _axis_angle_to_quaternion_wxyz(pose[:, :3]).astype(np.float64)
    if smpl_y_up:
        half = np.pi / 4.0
        y_to_z = np.asarray([np.cos(half), np.sin(half), 0.0, 0.0], dtype=np.float64)
        root = _quat_multiply_wxyz(y_to_z, root)
    # remove_smpl_base_rot(quat, w_last=False) == quat * conjugate(base_rot)
    base_conjugate = np.asarray([0.5, -0.5, -0.5, -0.5], dtype=np.float64)
    root = _quat_multiply_wxyz(root, base_conjugate)
    return _normalize_quaternions(root)


def _mujoco_forward_kinematics(
    *,
    joint_pos: Any,
    root_pos: Any,
    root_quat_wxyz: Any,
    model_path: str | Path,
    source_joint_order: Any = None,
    requested_body_order: Sequence[str] | None = None,
) -> tuple[Any, Any, tuple[str, ...]]:
    """Materialize body poses from G1 joint/root signals using MuJoCo.

    This helper is intentionally called only by :func:`normalize_sonic_motion`
    on the cold path.  It keeps XML parsing and model metadata out of the
    rollout step while allowing upstream motion-lib PKLs (which contain only
    ``dof``/``root_trans_offset``/``root_rot``) to become UniLab clips.
    """

    import numpy as np

    try:
        import mujoco as mujoco_module
    except ImportError as exc:  # pragma: no cover - depends on optional backend
        raise MotionManifestError(
            "MuJoCo is required for FK when body fields are absent; install mujoco or "
            "provide body_pos_w/body_quat_w explicitly"
        ) from exc
    mujoco: Any = mujoco_module
    model_file = Path(model_path).expanduser().resolve()
    if not model_file.is_file():
        raise MotionManifestError(f"FK model XML does not exist: {model_file}")
    try:
        model = mujoco.MjModel.from_xml_path(str(model_file))
    except Exception as exc:  # noqa: BLE001
        raise MotionManifestError(f"could not load FK model {model_file}: {exc}") from exc
    data = mujoco.MjData(model)
    qpos_values = np.asarray(joint_pos, dtype=np.float32)
    root_values = np.asarray(root_pos, dtype=np.float32)
    quat_values = np.asarray(root_quat_wxyz, dtype=np.float32)
    if qpos_values.ndim != 2 or root_values.shape != (qpos_values.shape[0], 3):
        raise MotionManifestError("FK inputs must have joint_pos=(T,J) and root_pos=(T,3)")
    if quat_values.shape != (qpos_values.shape[0], 4):
        raise MotionManifestError("FK root quaternion must have shape (T,4)")

    # Resolve source joint names to model hinge/slide qpos addresses.  Named
    # mappings are fail-closed; the positional fallback is only used for the
    # explicit ``joint_order='mj'`` convention or when dimensions match model.nu.
    joint_ids: list[int] = []
    source_names = (
        tuple(str(name) for name in source_joint_order)
        if isinstance(source_joint_order, (list, tuple))
        else None
    )
    model_joint_by_name: dict[str, int] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name:
            model_joint_by_name[str(name)] = joint_id
    actuator_joint_ids = [int(model.actuator(i).trnid[0]) for i in range(model.nu)]
    if source_names is not None:
        for name in source_names:
            normalized = name.removesuffix("_dof")
            matched_joint_id = model_joint_by_name.get(name, model_joint_by_name.get(normalized))
            if matched_joint_id is None:
                raise MotionManifestError(f"FK model has no joint named {name!r}")
            joint_ids.append(matched_joint_id)
    elif qpos_values.shape[1] == model.nu:
        joint_ids = actuator_joint_ids
    else:
        raise MotionManifestError(
            "FK requires source joint_order names when joint count does not match model.nu"
        )

    body_names_model: list[str] = []
    body_ids: list[int] = []
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        body_names_model.append(str(name) if name else f"body_{body_id}")
    if requested_body_order is None:
        body_ids = list(range(model.nbody))
        selected_body_names = tuple(body_names_model)
    else:
        for name in requested_body_order:
            try:
                body_ids.append(body_names_model.index(str(name)))
            except ValueError as exc:
                raise MotionManifestError(f"FK model has no body named {name!r}") from exc
        selected_body_names = tuple(str(name) for name in requested_body_order)

    body_pos = np.empty((qpos_values.shape[0], len(body_ids), 3), dtype=np.float32)
    body_quat = np.empty((qpos_values.shape[0], len(body_ids), 4), dtype=np.float32)
    for frame_index in range(qpos_values.shape[0]):
        qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
        if model.nq < 7:
            raise MotionManifestError("FK model must contain a free root (nq >= 7)")
        qpos[:3] = root_values[frame_index]
        qpos[3:7] = quat_values[frame_index]
        for joint_index, joint_id in enumerate(joint_ids):
            if joint_index >= qpos_values.shape[1]:
                break
            if model.jnt_type[joint_id] not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                raise MotionManifestError(
                    f"FK source joint {joint_index} maps to unsupported MuJoCo joint type"
                )
            qpos[int(model.jnt_qposadr[joint_id])] = qpos_values[frame_index, joint_index]
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        body_pos[frame_index] = np.asarray(data.xpos[body_ids], dtype=np.float32)
        body_quat[frame_index] = np.asarray(data.xquat[body_ids], dtype=np.float32)
    return body_pos, body_quat, selected_body_names


def normalize_sonic_motion(
    source: str | Path | Mapping[str, Any],
    *,
    source_fps: int | float | None = None,
    target_fps: int | float = 50,
    joint_order: Sequence[str] | None = None,
    body_order: Sequence[str] | None = None,
    aliases: Mapping[str, Sequence[str]] | None = None,
    clip_id: str | None = None,
    quaternion_order: str = "wxyz",
    fk_model_path: str | Path | None = None,
    derive_velocities: bool = True,
    smpl_y_up: bool = False,
) -> dict[str, Any]:
    """Normalize one SONIC source into UniLab's canonical motion fields.

    This is a cold-path converter.  It accepts a direct mapping or a path to
    ``.npz``, ``.npy``, ``.pkl`` and ``.joblib``.  The common SONIC motion-lib
    names (``root_trans_offset``, ``dof``, ``root_rot``) and UniLab names are
    resolved once here; the runtime only needs the normalized NPZ schema:

    ``fps``, ``joint_pos``, ``joint_vel``, ``body_pos_w``, ``body_quat_w``,
    ``body_lin_vel_w``, ``body_ang_vel_w`` and optional ``smpl_*`` fields.

    Positions are expected in metres and joint angles in radians.  No silent
    unit conversion is performed because guessing centimetres/degrees would
    make a release-parity run irreproducible.  ``root_rot`` from the upstream
    motion-lib convention is XYZW and is converted to the canonical WXYZ
    representation; all other quaternion aliases default to ``quaternion_order``.
    """

    import numpy as np

    if isinstance(quaternion_order, str):
        quaternion_order = quaternion_order.lower()
    if quaternion_order not in {"wxyz", "xyzw"}:
        raise MotionManifestError("quaternion_order must be 'wxyz' or 'xyzw'")
    target_rate = _scalar_number(target_fps, "target_fps")

    raw = _unwrap_motion_sequence(_load_motion_source(source), clip_id=clip_id)
    fps_value = None
    for name in _FPS_ALIASES:
        if name in raw and raw[name] is not None:
            fps_value = raw[name]
            break
    source_rate = _scalar_number(
        source_fps if source_fps is not None else fps_value,
        "source_fps",
    )

    resolved: dict[str, tuple[np.ndarray, str]] = {}
    frame_count: int | None = None
    for canonical in _MOTION_FIELD_ALIASES:
        value, source_name = _resolve_motion_field(raw, canonical, aliases)
        if value is None or source_name is None:
            continue
        array = _as_array(value, source_name)
        count = _validate_frame_axis(array, source_name, frame_count)
        frame_count = count if frame_count is None else frame_count
        resolved[canonical] = (array, source_name)

    if frame_count is None:
        raise MotionManifestError("motion source has no recognized frame-aligned fields")

    # Motion-lib entries store root translation/rotation separately.  Promote
    # them to body 0 so the canonical body arrays remain self-contained.
    root_pos = resolved.get("root_pos", (None, ""))[0]
    root_quat = resolved.get("root_quat", (None, ""))[0]
    body_pos = resolved.get("body_pos_w", (None, ""))[0]
    body_quat = resolved.get("body_quat_w", (None, ""))[0]
    if root_pos is not None:
        root_pos = np.asarray(root_pos, dtype=np.float32)
        if root_pos.ndim == 2 and root_pos.shape[1] == 3:
            pass
        elif root_pos.ndim == 3 and root_pos.shape[1:] == (1, 3):
            root_pos = root_pos[:, 0]
        else:
            raise MotionManifestError("root_pos must have shape (T,3)")
    if root_quat is not None:
        root_quat = np.asarray(root_quat, dtype=np.float32)
        if root_quat.ndim == 3 and root_quat.shape[1:] == (1, 4):
            root_quat = root_quat[:, 0]
        if root_quat.ndim != 2 or root_quat.shape[1] != 4:
            raise MotionManifestError("root_quat must have shape (T,4)")

    source_body_order = raw.get("body_order")
    if body_pos is None and fk_model_path is not None:
        # FK needs canonical root signals and joint positions, which are
        # validated below.  Defer the call until after quaternion conversion.
        pass
    elif body_pos is not None:
        if body_pos.ndim == 2 and body_pos.shape[1] == 3:
            body_pos = body_pos[:, None, :]
        if body_pos.ndim != 3 or body_pos.shape[-1] != 3:
            raise MotionManifestError("body_pos_w must have shape (T,B,3)")
    elif root_pos is not None:
        body_pos = root_pos[:, None, :]
    elif fk_model_path is None:
        raise MotionManifestError("motion source requires body_pos_w or root_pos")

    if body_quat is not None:
        if body_quat.ndim == 2 and body_quat.shape[1] == 4:
            body_quat = body_quat[:, None, :]
        if body_quat.ndim != 3 or body_quat.shape[-1] != 4:
            raise MotionManifestError("body_quat_w must have shape (T,B,4)")
    elif body_pos is not None:
        body_quat = np.zeros((*body_pos.shape[:2], 4), dtype=np.float32)
        body_quat[..., 0] = 1.0
    if root_pos is None and body_pos is not None:
        root_pos = body_pos[:, 0].copy()
    if root_quat is None and body_quat is not None:
        root_quat = body_quat[:, 0].copy()

    # Convert quaternion order before any interpolation or angular velocity
    # calculation.  ``root_rot`` is the upstream motion-lib XYZW alias.
    root_source_name = resolved.get("root_quat", (None, ""))[1]
    if root_pos is None:
        raise MotionManifestError("motion source requires root_pos or body_pos_w")
    if root_quat is not None and (root_source_name == "root_rot" or quaternion_order == "xyzw"):
        root_quat = root_quat[..., [3, 0, 1, 2]]
    body_source_name = resolved.get("body_quat_w", (None, ""))[1]
    body_order_source = raw.get("body_quat_order", raw.get("quaternion_order", quaternion_order))
    if (
        body_quat is not None
        and body_source_name not in {"body_quat_w"}
        and str(body_order_source).lower() == "xyzw"
    ):
        body_quat = body_quat[..., [3, 0, 1, 2]]
    if root_quat is None:
        root_quat = np.zeros((frame_count, 4), dtype=np.float32)
        root_quat[:, 0] = 1.0
    root_quat = _normalize_quaternions(root_quat)
    if body_quat is not None:
        body_quat = _normalize_quaternions(body_quat)

    # Ensure body 0 is exactly the root signal.  This avoids tiny source-file
    # discrepancies that otherwise make reset fixtures non-deterministic.
    # Joint positions are needed both for the canonical output and optional
    # MuJoCo FK, so resolve them before constructing missing body fields.

    joint_pair = resolved.get("joint_pos")
    if joint_pair is None:
        raise MotionManifestError("motion source requires joint_pos/dof_pos/dof")
    joint_pos = np.asarray(joint_pair[0], dtype=np.float32)
    if joint_pos.ndim != 2:
        raise MotionManifestError("joint_pos must have shape (T,J)")
    joint_vel_pair = resolved.get("joint_vel")
    joint_vel = (
        np.asarray(joint_vel_pair[0], dtype=np.float32) if joint_vel_pair is not None else None
    )
    if joint_vel is not None and (joint_vel.ndim != 2 or joint_vel.shape != joint_pos.shape):
        raise MotionManifestError("joint_vel must have the same shape as joint_pos")

    resampled_before_fk = False
    if body_pos is None and abs(source_rate - target_rate) > 1e-9:
        # SONIC evaluates FK after generalized coordinates reach the target
        # grid; post-FK body interpolation differs for articulated rotations.
        joint_pos = _resample_linear(joint_pos, source_rate, target_rate)
        root_pos = _resample_linear(root_pos, source_rate, target_rate)
        root_quat = _resample_quaternion(root_quat, source_rate, target_rate)
        resampled_before_fk = True

    if body_pos is None:
        if fk_model_path is None:  # pragma: no cover - guarded by earlier branch
            raise MotionManifestError("fk_model_path is required when body_pos_w is absent")
        body_pos, body_quat, fk_body_names = _mujoco_forward_kinematics(
            joint_pos=joint_pos,
            root_pos=root_pos,
            root_quat_wxyz=root_quat,
            model_path=fk_model_path,
            source_joint_order=raw.get("joint_order"),
            requested_body_order=body_order,
        )
        if body_order is None:
            body_order = fk_body_names
    elif body_quat is None:
        body_quat = np.zeros((*body_pos.shape[:2], 4), dtype=np.float32)
        body_quat[..., 0] = 1.0

    body_pos = np.asarray(body_pos, dtype=np.float32).copy()
    body_quat = np.asarray(body_quat, dtype=np.float32).copy()
    body_pos[:, 0] = root_pos
    body_quat[:, 0] = root_quat

    # Reorder named joints/bodies exactly once.  Symbolic values such as
    # ``joint_order='mj'`` are intentionally left untouched; the caller must
    # provide explicit names when a permutation is required.
    source_joint_order = raw.get("joint_order")
    if (
        joint_order is not None
        and isinstance(source_joint_order, (list, tuple))
        and len(source_joint_order) == joint_pos.shape[1]
    ):
        requested = tuple(str(name) for name in joint_order)
        source_names = tuple(str(name) for name in source_joint_order)
        if len(requested) != len(source_names) or set(requested) != set(source_names):
            raise MotionManifestError("joint_order does not contain the same names as the source")
        indices = [source_names.index(name) for name in requested]
        joint_pos = joint_pos[:, indices]
        if joint_vel is not None:
            joint_vel = joint_vel[:, indices]
    source_body_order = raw.get("body_order")
    if (
        body_order is not None
        and isinstance(source_body_order, (list, tuple))
        and len(source_body_order) == body_pos.shape[1]
    ):
        requested = tuple(str(name) for name in body_order)
        source_names = tuple(str(name) for name in source_body_order)
        if len(requested) != len(source_names) or set(requested) != set(source_names):
            raise MotionManifestError("body_order does not contain the same names as the source")
        indices = [source_names.index(name) for name in requested]
        body_pos = body_pos[:, indices]
        body_quat = body_quat[:, indices]
    elif body_order is not None and len(tuple(body_order)) != body_pos.shape[1]:
        raise MotionManifestError(
            "body_order length does not match body_pos_w; provide named source body_order "
            "when a permutation or subset is required"
        )
    body_pos[:, 0] = root_pos
    body_quat[:, 0] = root_quat

    # Resample all pose/position signals on a common time grid.  Velocities are
    # recomputed after resampling, avoiding a stale dt from the source corpus.
    if abs(source_rate - target_rate) > 1e-9 and not resampled_before_fk:
        joint_pos = _resample_linear(joint_pos, source_rate, target_rate)
        body_pos = _resample_linear(body_pos, source_rate, target_rate)
        body_quat = _resample_quaternion(body_quat, source_rate, target_rate)
        root_pos = body_pos[:, 0].copy()
        root_quat = body_quat[:, 0].copy()

    dt = 1.0 / target_rate
    if derive_velocities or joint_vel is None:
        joint_vel = _finite_difference(joint_pos, dt)
    elif abs(source_rate - target_rate) > 1e-9:
        joint_vel = _resample_linear(joint_vel, source_rate, target_rate)
    body_lin_pair = resolved.get("body_lin_vel_w")
    body_ang_pair = resolved.get("body_ang_vel_w")
    body_lin_vel = (
        _finite_difference(body_pos, dt)
        if derive_velocities or body_lin_pair is None
        else np.asarray(body_lin_pair[0], dtype=np.float32)
    )
    body_ang_vel = (
        _quaternion_angular_velocity(body_quat, dt)
        if derive_velocities or body_ang_pair is None
        else np.asarray(body_ang_pair[0], dtype=np.float32)
    )
    if not derive_velocities and abs(source_rate - target_rate) > 1e-9:
        if body_lin_pair is not None:
            body_lin_vel = _resample_linear(body_lin_vel, source_rate, target_rate)
        if body_ang_pair is not None:
            body_ang_vel = _resample_linear(body_ang_vel, source_rate, target_rate)
    if body_lin_vel.shape != body_pos.shape:
        raise MotionManifestError("body_lin_vel_w must have the same shape as body_pos_w")
    if body_ang_vel.shape != body_pos.shape:
        raise MotionManifestError("body_ang_vel_w must have shape (T,B,3)")

    output: dict[str, Any] = {
        "fps": np.asarray(int(round(target_rate)), dtype=np.int32),
        "root_pos": np.asarray(root_pos, dtype=np.float32),
        "root_quat": np.asarray(root_quat, dtype=np.float32),
        "root_lin_vel_w": np.asarray(body_lin_vel[:, 0], dtype=np.float32),
        "root_ang_vel_w": np.asarray(body_ang_vel[:, 0], dtype=np.float32),
        "joint_pos": np.asarray(joint_pos, dtype=np.float32),
        "joint_vel": np.asarray(joint_vel, dtype=np.float32),
        "body_pos_w": np.asarray(body_pos, dtype=np.float32),
        "body_quat_w": np.asarray(body_quat, dtype=np.float32),
        "body_lin_vel_w": np.asarray(body_lin_vel, dtype=np.float32),
        "body_ang_vel_w": np.asarray(body_ang_vel, dtype=np.float32),
    }

    # Optional SMPL signals are preserved under canonical names.  Flattened
    # ``(T,72)`` joints are expanded to ``(T,24,3)`` for the SONIC encoder.
    smpl_root_pair = resolved.get("smpl_root_quat_w")
    smpl_root_quat = None
    if smpl_root_pair is not None and resolved.get("smpl_pose") is None:
        smpl_root_quat = np.asarray(smpl_root_pair[0], dtype=np.float32)
        if smpl_root_quat.ndim != 2 or smpl_root_quat.shape[1] != 4:
            raise MotionManifestError("smpl_root_quat_w must have shape (T,4)")
        raw_smpl_order = raw.get("smpl_root_quat_order", raw.get("quaternion_order"))
        if str(raw_smpl_order).lower() == "xyzw":
            smpl_root_quat = smpl_root_quat[..., [3, 0, 1, 2]]
        smpl_root_quat = _normalize_quaternions(smpl_root_quat)
        if abs(source_rate - target_rate) > 1e-9:
            smpl_root_quat = _resample_quaternion(smpl_root_quat, source_rate, target_rate)

    for canonical in ("smpl_joints", "smpl_pose", "smpl_transl"):
        pair = resolved.get(canonical)
        if pair is None:
            continue
        value = np.asarray(pair[0], dtype=np.float32)
        if value.shape[0] != frame_count:
            raise MotionManifestError(f"{canonical} frame count does not match robot fields")
        if canonical == "smpl_pose":
            if value.ndim != 2 or value.shape[1] % 3 != 0:
                raise MotionManifestError("smpl_pose must have shape (T,J*3)")
            value = value.copy()
            value[:, -6:] = 0.0
        if canonical == "smpl_joints" and value.ndim == 2:
            if value.shape[1] % 3 != 0:
                raise MotionManifestError("flattened smpl_joints must have a multiple-of-3 width")
            value = value.reshape(value.shape[0], value.shape[1] // 3, 3)
        if abs(source_rate - target_rate) > 1e-9:
            value = _resample_linear(value, source_rate, target_rate)
        output[canonical] = value
    # SONIC derives root orientation from the target-grid pose.  Keep an
    # explicit quaternion only as the fallback for sources without pose_aa.
    if "smpl_pose" in output:
        smpl_root_quat = _derive_smpl_root_quaternion(output["smpl_pose"], smpl_y_up=smpl_y_up)
    if smpl_root_quat is not None:
        output["smpl_root_quat_w"] = np.asarray(smpl_root_quat, dtype=np.float32)

    # Preserve an action stream when present; it is useful for motion-lib
    # pretraining but is not required by the standard tracking loader.
    action_value, _ = _resolve_motion_field(raw, "action", aliases)
    if action_value is not None:
        action = _as_array(action_value, "action")
        _validate_frame_axis(action, "action", frame_count)
        output["action"] = _resample_linear(action, source_rate, target_rate)
    return output


def convert_sonic_motion(
    source: str | Path | Mapping[str, Any],
    output_path: str | Path,
    *,
    source_fps: int | float | None = None,
    target_fps: int | float = 50,
    joint_order: Sequence[str] | None = None,
    body_order: Sequence[str] | None = None,
    aliases: Mapping[str, Sequence[str]] | None = None,
    clip_id: str | None = None,
    quaternion_order: str = "wxyz",
    fk_model_path: str | Path | None = None,
    derive_velocities: bool = True,
    smpl_y_up: bool = False,
    overwrite: bool = False,
    compressed: bool = True,
) -> MotionConversionReport:
    """Write one normalized SONIC NPZ clip and return its checksum report."""

    import tempfile

    import numpy as np

    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() != ".npz":
        raise MotionManifestError("normalized motion output must use the .npz suffix")
    if destination.exists() and not overwrite:
        raise MotionManifestError(f"normalized motion output already exists: {destination}")
    arrays = normalize_sonic_motion(
        source,
        source_fps=source_fps,
        target_fps=target_fps,
        joint_order=joint_order,
        body_order=body_order,
        aliases=aliases,
        clip_id=clip_id,
        quaternion_order=quaternion_order,
        fk_model_path=fk_model_path,
        derive_velocities=derive_velocities,
        smpl_y_up=smpl_y_up,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    # ``np.savez`` appends a suffix when given a string, so use an open file
    # descriptor in a temporary sibling and atomically replace the destination.
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".npz", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        saver = np.savez_compressed if compressed else np.savez
        saver(temporary, **arrays)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    checksum = sha256_file(destination)
    return MotionConversionReport(
        output_path=destination,
        checksum=checksum,
        fps=int(np.asarray(arrays["fps"]).item()),
        num_frames=int(arrays["joint_pos"].shape[0]),
        fields=tuple(name for name in arrays if name != "fps"),
    )


# Explicit aliases make the API discoverable for callers that use the
# materializer terminology while retaining one implementation/contract.
normalize_motion_clip = normalize_sonic_motion
convert_motion_clip = convert_sonic_motion


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MotionManifestError(f"{location} must be an object")
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MotionManifestError(f"{location} must be a non-empty string")
    return value


def _order(value: Any, location: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise MotionManifestError(f"{location} must be a non-empty list")
    result = tuple(_string(item, f"{location}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise MotionManifestError(f"{location} must not contain duplicate names")
    return result


def _shape(value: Any, location: str) -> tuple[int | str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise MotionManifestError(f"{location} must be a non-empty list")
    result: list[int | str] = []
    for index, item in enumerate(value):
        item_location = f"{location}[{index}]"
        if item is None:
            result.append(-1)
        elif isinstance(item, bool):
            raise MotionManifestError(f"{item_location} must be an integer or supported symbol")
        elif isinstance(item, int):
            if item < -1:
                raise MotionManifestError(f"{item_location} must be >= -1")
            result.append(item)
        elif isinstance(item, str) and item in _SHAPE_SYMBOLS:
            result.append(item)
        else:
            raise MotionManifestError(
                f"{item_location} must be an integer or one of {sorted(_SHAPE_SYMBOLS)}"
            )
    return tuple(result)


@dataclass(frozen=True)
class MotionFieldSpec:
    """Shape and dtype contract for one array stored by each clip."""

    name: str
    shape: tuple[int | str, ...]
    dtype: str

    @classmethod
    def from_dict(cls, value: Any, location: str) -> "MotionFieldSpec":
        data = _mapping(value, location)
        allowed = {"name", "shape", "dtype"}
        unknown = set(data) - allowed
        if unknown:
            raise MotionManifestError(f"{location} has unknown keys: {sorted(unknown)}")
        name = _string(data.get("name"), f"{location}.name")
        dtype = _string(data.get("dtype"), f"{location}.dtype")
        try:
            import numpy as np

            np.dtype(dtype)
        except Exception as exc:  # noqa: BLE001
            raise MotionManifestError(f"{location}.dtype is invalid: {dtype!r}") from exc
        return cls(name=name, shape=_shape(data.get("shape"), f"{location}.shape"), dtype=dtype)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}


MotionField = MotionFieldSpec


@dataclass(frozen=True)
class MotionClip:
    """Metadata for one immutable motion clip or array shard."""

    id: str
    path: str
    checksum: str
    fps: int
    num_frames: int
    joint_order: tuple[str, ...] | None = None
    body_order: tuple[str, ...] | None = None

    @property
    def sha256(self) -> str:
        return self.checksum

    @classmethod
    def from_dict(cls, value: Any, location: str) -> "MotionClip":
        data = _mapping(value, location)
        allowed = {
            "id",
            "path",
            "sha256",
            "checksum",
            "fps",
            "num_frames",
            "joint_order",
            "body_order",
        }
        unknown = set(data) - allowed
        if unknown:
            raise MotionManifestError(f"{location} has unknown keys: {sorted(unknown)}")
        checksum_value = data.get("sha256", data.get("checksum"))
        if "sha256" in data and "checksum" in data and data["sha256"] != data["checksum"]:
            raise MotionManifestError(f"{location} sha256 and checksum disagree")
        checksum = _string(checksum_value, f"{location}.sha256")
        if _SHA256_RE.fullmatch(checksum) is None:
            raise MotionManifestError(f"{location}.sha256 must be 64 hexadecimal characters")
        fps = data.get("fps")
        num_frames = data.get("num_frames")
        if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
            raise MotionManifestError(f"{location}.fps must be a positive integer")
        if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
            raise MotionManifestError(f"{location}.num_frames must be a positive integer")
        return cls(
            id=_string(data.get("id"), f"{location}.id"),
            path=_string(data.get("path"), f"{location}.path"),
            checksum=checksum.lower(),
            fps=fps,
            num_frames=num_frames,
            joint_order=(
                _order(data["joint_order"], f"{location}.joint_order")
                if "joint_order" in data
                else None
            ),
            body_order=(
                _order(data["body_order"], f"{location}.body_order")
                if "body_order" in data
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "path": self.path,
            "sha256": self.checksum,
            "fps": self.fps,
            "num_frames": self.num_frames,
        }
        if self.joint_order is not None:
            result["joint_order"] = list(self.joint_order)
        if self.body_order is not None:
            result["body_order"] = list(self.body_order)
        return result


@dataclass(frozen=True)
class MotionManifest:
    """Parsed SONIC motion manifest and its immutable schema metadata."""

    version: int
    joint_order: tuple[str, ...]
    body_order: tuple[str, ...]
    fields: tuple[MotionFieldSpec, ...]
    clips: tuple[MotionClip, ...]
    schema: str = MANIFEST_SCHEMA
    metadata: Mapping[str, Any] = field(default_factory=dict)
    manifest_path: Path | None = field(default=None, compare=False, repr=False)

    @classmethod
    def from_dict(cls, value: Any, *, manifest_path: str | Path | None = None) -> "MotionManifest":
        data = _mapping(value, "manifest")
        allowed = {
            "schema",
            "version",
            "joint_order",
            "body_order",
            "fields",
            "clips",
            "metadata",
        }
        unknown = set(data) - allowed
        if unknown:
            raise MotionManifestError(f"manifest has unknown keys: {sorted(unknown)}")
        version = data.get("version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise MotionManifestError("manifest.version must be an integer")
        if version != MANIFEST_VERSION:
            raise MotionManifestError(
                f"unsupported manifest.version={version}; expected {MANIFEST_VERSION}"
            )
        schema = data.get("schema", MANIFEST_SCHEMA)
        if schema != MANIFEST_SCHEMA:
            raise MotionManifestError(
                f"unsupported manifest.schema={schema!r}; expected {MANIFEST_SCHEMA!r}"
            )
        fields_value = data.get("fields")
        if isinstance(fields_value, Mapping):
            fields_value = [dict(spec, name=name) for name, spec in fields_value.items()]
        if not isinstance(fields_value, (list, tuple)) or not fields_value:
            raise MotionManifestError("manifest.fields must be a non-empty list or object")
        fields = tuple(
            MotionFieldSpec.from_dict(item, f"manifest.fields[{index}]")
            for index, item in enumerate(fields_value)
        )
        field_names = tuple(item.name for item in fields)
        if len(set(field_names)) != len(field_names):
            raise MotionManifestError("manifest.fields must not contain duplicate names")
        clips_value = data.get("clips")
        if not isinstance(clips_value, (list, tuple)) or not clips_value:
            raise MotionManifestError("manifest.clips must be a non-empty list")
        clips = tuple(
            MotionClip.from_dict(item, f"manifest.clips[{index}]")
            for index, item in enumerate(clips_value)
        )
        clip_ids = tuple(item.id for item in clips)
        if len(set(clip_ids)) != len(clip_ids):
            raise MotionManifestError("manifest.clips ids must be unique")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise MotionManifestError("manifest.metadata must be an object")
        parsed_path = Path(manifest_path).expanduser().resolve() if manifest_path else None
        result = cls(
            version=version,
            schema=schema,
            joint_order=_order(data.get("joint_order"), "manifest.joint_order"),
            body_order=_order(data.get("body_order"), "manifest.body_order"),
            fields=fields,
            clips=clips,
            metadata=dict(metadata),
            manifest_path=parsed_path,
        )
        _validate_order_contract(result)
        return result

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "version": self.version,
            "joint_order": list(self.joint_order),
            "body_order": list(self.body_order),
            "fields": [item.to_dict() for item in self.fields],
            "clips": [item.to_dict() for item in self.clips],
        }
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


def _validate_order_contract(manifest: MotionManifest) -> None:
    for clip in manifest.clips:
        if clip.joint_order is not None and clip.joint_order != manifest.joint_order:
            raise MotionManifestError(f"clip {clip.id!r} joint_order differs from manifest")
        if clip.body_order is not None and clip.body_order != manifest.body_order:
            raise MotionManifestError(f"clip {clip.id!r} body_order differs from manifest")
    field_names = {item.name for item in manifest.fields}
    for name in set(("joint_pos", "joint_vel", "joint_acc", "dof_pos", "dof_vel")) & field_names:
        spec = next(item for item in manifest.fields if item.name == name)
        if len(spec.shape) < 2 or spec.shape[-1] not in {
            -1,
            "*",
            "num_joints",
            len(manifest.joint_order),
        }:
            raise MotionManifestError(
                f"field {name!r} must expose the manifest joint order on its last axis"
            )
    for name in (
        set(("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")) & field_names
    ):
        spec = next(item for item in manifest.fields if item.name == name)
        if len(spec.shape) < 3 or spec.shape[-2] not in {
            -1,
            "*",
            "num_bodies",
            len(manifest.body_order),
        }:
            raise MotionManifestError(
                f"field {name!r} must expose the manifest body order on its penultimate axis"
            )


def resolve_manifest_clip_path(manifest_path: str | Path, clip_path: str) -> Path:
    """Resolve a clip relative to the manifest while rejecting path escapes."""

    manifest = Path(manifest_path).expanduser().resolve()
    raw = _string(clip_path, "clip.path")
    if "\x00" in raw:
        raise MotionManifestError("clip.path must not contain NUL bytes")
    windows_path = PureWindowsPath(raw)
    if Path(raw).is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise MotionManifestError(f"clip.path must be relative: {raw!r}")
    if any(part == ".." for part in windows_path.parts):
        raise MotionManifestError(f"clip.path must not contain parent traversal: {raw!r}")
    root = manifest.parent
    resolved = (root / raw).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MotionManifestError(f"clip.path escapes manifest directory: {raw!r}") from exc
    return resolved


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Compute a file SHA256 digest on the cold path."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_dimension(
    dimension: int | str, clip: MotionClip, manifest: MotionManifest
) -> int | None:
    if dimension == -1 or dimension == "*":
        return None
    if dimension == "num_frames":
        return clip.num_frames
    if dimension == "num_joints":
        return len(manifest.joint_order)
    if dimension == "num_bodies":
        return len(manifest.body_order)
    return int(dimension)


def _validate_array_contract(path: Path, clip: MotionClip, manifest: MotionManifest) -> None:
    import numpy as np

    expected = {field.name: field for field in manifest.fields}
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
            _compare_arrays(arrays, expected, clip, manifest)
        return
    if suffix == ".npy":
        if len(expected) != 1:
            raise MotionManifestError("a .npy clip must declare exactly one field")
        arrays = {next(iter(expected)): np.load(path, mmap_mode="r", allow_pickle=False)}
        _compare_arrays(arrays, expected, clip, manifest)
        return
    raise MotionManifestError(
        f"clip {clip.id!r} has unsupported array format {path.suffix!r}; expected .npz or .npy"
    )


def _compare_arrays(
    arrays: Mapping[str, Any],
    expected: Mapping[str, MotionFieldSpec],
    clip: MotionClip,
    manifest: MotionManifest,
) -> None:
    # ``fps`` is clip metadata rather than a frame-aligned field.  It remains
    # in the NPZ for MotionLoader compatibility but is intentionally omitted
    # from ``manifest.fields``.
    metadata_names = {"fps"}
    unexpected = set(arrays) - set(expected) - metadata_names
    missing = sorted(set(expected) - set(arrays))
    if missing or unexpected:
        extra = sorted(unexpected)
        raise MotionManifestError(
            f"clip {clip.id!r} fields disagree; missing={missing}, extra={extra}"
        )
    if "fps" in arrays:
        fps_array = arrays["fps"]
        if getattr(fps_array, "ndim", None) != 0:
            raise MotionManifestError(f"clip {clip.id!r} metadata 'fps' must be scalar")
    import numpy as np

    for name, spec in expected.items():
        array = arrays[name]
        resolved_shape = tuple(_resolve_dimension(item, clip, manifest) for item in spec.shape)
        if len(resolved_shape) != array.ndim or any(
            dimension is not None and dimension != actual
            for dimension, actual in zip(resolved_shape, array.shape)
        ):
            raise MotionManifestError(
                f"clip {clip.id!r} field {name!r} shape {tuple(array.shape)} "
                f"does not match {spec.shape}"
            )
        try:
            expected_dtype = np.dtype(spec.dtype)
        except Exception as exc:  # noqa: BLE001
            raise MotionManifestError(f"field {name!r} has invalid dtype {spec.dtype!r}") from exc
        if np.dtype(array.dtype) != expected_dtype:
            raise MotionManifestError(
                f"clip {clip.id!r} field {name!r} dtype {array.dtype} does not match {spec.dtype}"
            )


def load_motion_manifest(path: str | Path) -> MotionManifest:
    """Read and schema-validate a manifest without loading clip arrays."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise MotionManifestError(f"manifest file does not exist: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise MotionManifestError(f"could not read manifest {manifest_path}: {exc}") from exc
    manifest = MotionManifest.from_dict(data, manifest_path=manifest_path)
    for clip in manifest.clips:
        resolve_manifest_clip_path(manifest_path, clip.path)
    return manifest


def validate_motion_manifest(
    manifest: MotionManifest | Mapping[str, Any] | str | Path,
    *,
    manifest_path: str | Path | None = None,
    check_files: bool = False,
    verify_checksums: bool = False,
    verify_shapes: bool = False,
    expected_joint_order: Sequence[str] | None = None,
    expected_body_order: Sequence[str] | None = None,
) -> MotionManifest:
    """Validate schema and optionally perform file, hash, and array checks."""

    if isinstance(manifest, (str, Path)):
        parsed = load_motion_manifest(manifest)
    elif isinstance(manifest, MotionManifest):
        parsed = manifest
    else:
        parsed = MotionManifest.from_dict(manifest, manifest_path=manifest_path)
    if expected_joint_order is not None and parsed.joint_order != tuple(expected_joint_order):
        raise MotionManifestError("manifest joint_order does not match the expected order")
    if expected_body_order is not None and parsed.body_order != tuple(expected_body_order):
        raise MotionManifestError("manifest body_order does not match the expected order")
    if check_files or verify_checksums or verify_shapes:
        source = manifest_path or parsed.manifest_path
        if source is None:
            raise MotionManifestError("manifest_path is required for file preflight")
        source = Path(source).expanduser().resolve()
        for clip in parsed.clips:
            clip_path = resolve_manifest_clip_path(source, clip.path)
            if not clip_path.is_file():
                raise MotionManifestError(f"clip {clip.id!r} does not exist: {clip_path}")
            if verify_checksums and sha256_file(clip_path) != clip.checksum:
                raise MotionManifestError(f"clip {clip.id!r} SHA256 checksum mismatch")
            if verify_shapes:
                _validate_array_contract(clip_path, clip, parsed)
    return parsed


def preflight_motion_manifest(
    manifest: MotionManifest | Mapping[str, Any] | str | Path,
    *,
    manifest_path: str | Path | None = None,
    verify_checksums: bool = True,
    verify_shapes: bool = True,
    expected_joint_order: Sequence[str] | None = None,
    expected_body_order: Sequence[str] | None = None,
) -> MotionManifest:
    """Fail-closed cold-path preflight for all clips in a manifest."""

    return validate_motion_manifest(
        manifest,
        manifest_path=manifest_path,
        check_files=True,
        verify_checksums=verify_checksums,
        verify_shapes=verify_shapes,
        expected_joint_order=expected_joint_order,
        expected_body_order=expected_body_order,
    )


def materialize_motion_store(
    source_files: Sequence[str | Path],
    output_dir: str | Path,
    *,
    fps: int,
    joint_order: Sequence[str],
    body_order: Sequence[str],
    overwrite: bool = False,
    copy_mode: str = "copy",
) -> MotionMaterializationReport:
    """Build a versioned, immutable NPZ store and manifest on the cold path.

    This materializer accepts normalized NPZ clips only.  Use
    :func:`convert_sonic_motion` first for SONIC PKL/SMPL sources; it performs
    one-time alias/FK/reordering/resampling work and writes the canonical NPZ.
    A scalar ``fps`` array is retained in each clip as metadata (and checked
    against ``fps``) but is not treated as a frame-aligned manifest field.
    The resulting store gives every rank the same checksummed shard contract
    and keeps PKL/XML parsing out of ``reset``/``step``.

    ``source_files`` are read once to infer field shapes/dtypes and frame
    counts.  The resulting files are copied (or hard-linked) below
    ``output_dir/clips`` and never modified in place.
    """

    if isinstance(source_files, (str, bytes, Path)):
        raise MotionManifestError("source_files must be a non-empty sequence, not one path")
    sources = tuple(Path(path).expanduser().resolve() for path in source_files)
    if not sources:
        raise MotionManifestError("source_files must not be empty")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise MotionManifestError("fps must be a positive integer")
    joint_names = _order(list(joint_order), "joint_order")
    body_names = _order(list(body_order), "body_order")
    if copy_mode not in {"copy", "hardlink"}:
        raise MotionManifestError("copy_mode must be 'copy' or 'hardlink'")

    import numpy as np

    for source in sources:
        if not source.is_file():
            raise MotionManifestError(f"source motion clip does not exist: {source}")
        if source.suffix.lower() != ".npz":
            raise MotionManifestError(f"source motion clip must be .npz, got {source.name!r}")

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise MotionManifestError(
            f"materialization output directory is not empty: {destination}; "
            "pass overwrite=True to rebuild it"
        )
    clips_dir = destination / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    field_specs: dict[str, MotionFieldSpec] | None = None
    clips: list[MotionClip] = []
    total_frames = 0
    total_bytes = 0
    used_names: set[str] = set()
    for index, source in enumerate(sources):
        with np.load(source, allow_pickle=False) as archive:
            if not archive.files:
                raise MotionManifestError(f"source clip has no arrays: {source}")
            arrays = {name: archive[name] for name in archive.files}
            frame_arrays = {name: array for name, array in arrays.items() if name != "fps"}
            if not frame_arrays:
                raise MotionManifestError(f"source clip has no frame-aligned arrays: {source}")
            if "fps" in arrays:
                fps_value = np.asarray(arrays["fps"])
                if fps_value.ndim != 0:
                    raise MotionManifestError(
                        f"source clip metadata 'fps' must be scalar: {source}"
                    )
                try:
                    source_fps_value = float(fps_value)
                except (TypeError, ValueError) as exc:
                    raise MotionManifestError(
                        f"source clip metadata 'fps' is invalid: {source}"
                    ) from exc
                if abs(source_fps_value - fps) > 1e-6:
                    raise MotionManifestError(
                        f"source clip metadata fps={source_fps_value:g} disagrees with requested fps={fps}: {source}"
                    )
            num_frames = int(next(iter(frame_arrays.values())).shape[0])
            if num_frames <= 0:
                raise MotionManifestError(f"source clip has no frames: {source}")
            if any(
                array.ndim == 0 or int(array.shape[0]) != num_frames
                for array in frame_arrays.values()
            ):
                raise MotionManifestError(f"source clip arrays do not share a frame axis: {source}")
            current_specs = {
                name: MotionFieldSpec(
                    name=name,
                    shape=("num_frames", *tuple(int(extent) for extent in array.shape[1:])),
                    dtype=str(array.dtype),
                )
                for name, array in frame_arrays.items()
            }
            if field_specs is None:
                field_specs = current_specs
            elif {name: spec.to_dict() for name, spec in current_specs.items()} != {
                name: spec.to_dict() for name, spec in field_specs.items()
            }:
                raise MotionManifestError(
                    f"source clip fields/shapes/dtypes disagree with the first clip: {source}"
                )

        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem).strip("._") or "clip"
        name = f"{index:06d}_{stem}.npz"
        while name in used_names:
            name = f"{index:06d}_{stem}_{len(used_names)}.npz"
        used_names.add(name)
        target = clips_dir / name
        if target.exists() and overwrite:
            target.unlink()
        if copy_mode == "hardlink":
            try:
                target.hardlink_to(source)
            except OSError:
                shutil.copy2(source, target)
        else:
            shutil.copy2(source, target)
        checksum = sha256_file(target)
        clip_id = f"clip_{index:06d}"
        clips.append(
            MotionClip(
                id=clip_id,
                path=str(target.relative_to(destination)),
                checksum=checksum,
                fps=fps,
                num_frames=num_frames,
                joint_order=joint_names,
                body_order=body_names,
            )
        )
        total_frames += num_frames
        total_bytes += target.stat().st_size

    assert field_specs is not None  # guarded by the non-empty sources check.
    manifest = MotionManifest(
        version=MANIFEST_VERSION,
        schema=MANIFEST_SCHEMA,
        joint_order=joint_names,
        body_order=body_names,
        fields=tuple(field_specs.values()),
        clips=tuple(clips),
        metadata={"materializer": "unilab.sonic_motion", "source_count": len(sources)},
        manifest_path=destination / "manifest.json",
    )
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Re-open through the strict parser so the artifact on disk is the exact
    # contract consumed by worker ranks.
    preflight_motion_manifest(manifest_path, verify_checksums=True, verify_shapes=True)
    return MotionMaterializationReport(
        manifest_path=manifest_path,
        clip_count=len(clips),
        total_frames=total_frames,
        total_bytes=total_bytes,
    )


def _index_paired_source_root(root: str | Path, *, label: str) -> dict[str, Path]:
    source_root = Path(root).expanduser().resolve()
    if not source_root.is_dir():
        raise MotionManifestError(f"{label} source root is not a directory: {source_root}")
    paths_by_key: dict[str, list[Path]] = {}
    for path in source_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _PAIR_SOURCE_SUFFIXES:
            continue
        if path.stem == "metadata":
            continue
        paths_by_key.setdefault(path.stem, []).append(path.resolve())
    duplicates = {key: sorted(paths) for key, paths in paths_by_key.items() if len(paths) > 1}
    if duplicates:
        summary = "; ".join(
            f"{key}={','.join(str(path) for path in paths)}"
            for key, paths in sorted(duplicates.items())
        )
        raise MotionManifestError(f"duplicate {label} basename(s): {summary}")
    if not paths_by_key:
        suffixes = ", ".join(sorted(_PAIR_SOURCE_SUFFIXES))
        raise MotionManifestError(
            f"{label} source root has no supported motion files ({suffixes}): {source_root}"
        )
    return {key: paths[0] for key, paths in paths_by_key.items()}


def _summarize_keys(keys: Sequence[str], *, limit: int = 8) -> str:
    ordered = sorted(keys)
    suffix = f", ... (+{len(ordered) - limit})" if len(ordered) > limit else ""
    return ", ".join(ordered[:limit]) + suffix


def _paired_source_fps(
    source: Mapping[str, Any],
    *,
    pair_key: str,
    label: str,
    fallback: int | float | None,
) -> float:
    values = [
        (name, _scalar_number(source[name], f"pair {pair_key!r} {label}.{name}"))
        for name in _FPS_ALIASES
        if name in source and source[name] is not None
    ]
    if not values:
        if fallback is None:
            raise MotionManifestError(
                f"pair {pair_key!r} {label} source has no fps metadata; pass source_fps"
            )
        return _scalar_number(fallback, "source_fps")
    first_name, first_value = values[0]
    for name, value in values[1:]:
        if abs(value - first_value) > 1.0e-9:
            raise MotionManifestError(
                f"pair {pair_key!r} {label} fps aliases disagree: "
                f"{first_name}={first_value:g}, {name}={value:g}"
            )
    return first_value


def _unique_paired_field(
    source: Mapping[str, Any],
    *,
    pair_key: str,
    canonical: str,
    aliases: Sequence[str],
) -> Any | None:
    names = tuple(dict.fromkeys((canonical, *aliases)))
    present = [name for name in names if name in source and source[name] is not None]
    if len(present) > 1:
        raise MotionManifestError(
            f"pair {pair_key!r} SMPL field {canonical!r} is ambiguous: {present}"
        )
    if not present:
        return None
    return _as_array(source[present[0]], f"pair {pair_key!r} SMPL.{present[0]}")


def _normalize_paired_smpl_fields(
    fields: Mapping[str, Any],
    *,
    source_fps: float,
    target_fps: float,
    smpl_y_up: bool,
) -> dict[str, Any]:
    """Resample one SMPL source directly onto its target-rate time grid."""

    import numpy as np

    result: dict[str, Any] = {}
    smpl_root_quat = fields.get("smpl_root_quat_w")
    if smpl_root_quat is not None and fields.get("smpl_pose") is None:
        smpl_root_quat = np.asarray(smpl_root_quat, dtype=np.float32)
        if smpl_root_quat.ndim != 2 or smpl_root_quat.shape[1] != 4:
            raise MotionManifestError("smpl_root_quat_w must have shape (T,4)")
        smpl_root_quat = _normalize_quaternions(smpl_root_quat)
        if abs(source_fps - target_fps) > 1.0e-9:
            smpl_root_quat = _resample_quaternion(smpl_root_quat, source_fps, target_fps)

    for canonical in ("smpl_joints", "smpl_pose", "smpl_transl"):
        value = fields.get(canonical)
        if value is None:
            continue
        value = np.asarray(value, dtype=np.float32)
        if canonical == "smpl_pose" and (value.ndim != 2 or value.shape[1] % 3 != 0):
            raise MotionManifestError("smpl_pose must have shape (T,J*3)")
        if canonical == "smpl_pose":
            value = value.copy()
            value[:, -6:] = 0.0
        if canonical == "smpl_joints" and value.ndim == 2:
            if value.shape[1] % 3 != 0:
                raise MotionManifestError("flattened smpl_joints must have a multiple-of-3 width")
            value = value.reshape(value.shape[0], value.shape[1] // 3, 3)
        if abs(source_fps - target_fps) > 1.0e-9:
            value = _resample_linear(value, source_fps, target_fps)
        else:
            value = value.copy()
        result[canonical] = value
    # The upstream policy observes the root derived from its resampled pose,
    # even if a source happens to carry a separate quaternion field.
    if "smpl_pose" in result:
        smpl_root_quat = _derive_smpl_root_quaternion(result["smpl_pose"], smpl_y_up=smpl_y_up)
    if smpl_root_quat is not None:
        result["smpl_root_quat_w"] = np.asarray(smpl_root_quat, dtype=np.float32)
    return result


def _merge_paired_motion_source(
    robot_path: Path,
    smpl_path: Path,
    *,
    pair_key: str,
    source_fps: int | float | None,
    target_fps: int | float,
    joint_order: Sequence[str],
    body_order: Sequence[str],
    quaternion_order: str,
    fk_model_path: str | Path | None,
    derive_velocities: bool,
    smpl_y_up: bool,
) -> tuple[dict[str, Any], float]:
    try:
        robot = _unwrap_motion_sequence(_load_motion_source(robot_path))
    except MotionManifestError as exc:
        raise MotionManifestError(f"pair {pair_key!r} robot source is invalid: {exc}") from exc
    try:
        smpl = _unwrap_motion_sequence(_load_motion_source(smpl_path))
    except MotionManifestError as exc:
        raise MotionManifestError(f"pair {pair_key!r} SMPL source is invalid: {exc}") from exc

    robot_rate = _paired_source_fps(robot, pair_key=pair_key, label="robot", fallback=source_fps)
    smpl_rate = _paired_source_fps(smpl, pair_key=pair_key, label="SMPL", fallback=source_fps)
    target_rate = _scalar_number(target_fps, "target_fps")

    robot_joint_pos, robot_joint_name = _resolve_motion_field(robot, "joint_pos", None)
    if robot_joint_pos is None or robot_joint_name is None:
        raise MotionManifestError(f"pair {pair_key!r} robot source has no joint_pos/dof field")
    robot_frames = _validate_frame_axis(
        _as_array(robot_joint_pos, robot_joint_name), robot_joint_name, None
    )
    smpl_aliases = {
        "smpl_joints": _MOTION_FIELD_ALIASES["smpl_joints"],
        "smpl_pose": (*_MOTION_FIELD_ALIASES["smpl_pose"], "pose_aa"),
        "smpl_transl": (*_MOTION_FIELD_ALIASES["smpl_transl"], "transl"),
        "smpl_root_quat_w": _MOTION_FIELD_ALIASES["smpl_root_quat_w"],
    }
    smpl_fields = {
        name: value
        for name, aliases in smpl_aliases.items()
        if (value := _unique_paired_field(smpl, pair_key=pair_key, canonical=name, aliases=aliases))
        is not None
    }
    if "smpl_joints" not in smpl_fields:
        raise MotionManifestError(f"pair {pair_key!r} SMPL source has no smpl_joints field")
    if "smpl_pose" not in smpl_fields and "smpl_root_quat_w" not in smpl_fields:
        raise MotionManifestError(
            f"pair {pair_key!r} SMPL source needs pose_aa/smpl_pose or smpl_root_quat_w"
        )
    smpl_frames: int | None = None
    for name, value in smpl_fields.items():
        smpl_frames = _validate_frame_axis(value, name, smpl_frames)
    assert smpl_frames is not None

    # Normalize independently sampled sources from their own FPS; routing SMPL
    # through the robot rate would add a lossy official-corpus 50->30->50 trip.
    robot_source = dict(robot)
    for canonical in smpl_aliases:
        for alias in (canonical, *_MOTION_FIELD_ALIASES.get(canonical, ())):
            robot_source.pop(alias, None)
    normalized_robot = normalize_sonic_motion(
        robot_source,
        source_fps=robot_rate,
        target_fps=target_rate,
        joint_order=joint_order,
        body_order=body_order,
        quaternion_order=quaternion_order,
        fk_model_path=fk_model_path,
        derive_velocities=derive_velocities,
        smpl_y_up=smpl_y_up,
    )
    normalized_smpl = _normalize_paired_smpl_fields(
        smpl_fields,
        source_fps=smpl_rate,
        target_fps=target_rate,
        smpl_y_up=smpl_y_up,
    )
    robot_target_frames = int(normalized_robot["joint_pos"].shape[0])
    smpl_target_frames = int(normalized_smpl["smpl_joints"].shape[0])
    if smpl_target_frames != robot_target_frames:
        robot_duration = (robot_frames - 1) / robot_rate
        smpl_duration = (smpl_frames - 1) / smpl_rate
        raise MotionManifestError(
            f"pair {pair_key!r} duration/target-grid mismatch at {target_rate:g} fps: "
            f"robot={robot_frames}@{robot_rate:g} ({robot_duration:g}s)"
            f"->{robot_target_frames}, SMPL={smpl_frames}@{smpl_rate:g} "
            f"({smpl_duration:g}s)->{smpl_target_frames}"
        )

    normalized_robot.update(normalized_smpl)
    return normalized_robot, target_rate


def _publish_staged_motion_store(staging: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        os.replace(staging, destination)
        return
    if destination.is_symlink() or not destination.is_dir():
        raise MotionManifestError(
            f"materialization output is not a regular directory: {destination}"
        )
    if any(destination.iterdir()) and not overwrite:
        raise MotionManifestError(
            f"materialization output directory is not empty: {destination}; "
            "pass overwrite=True to rebuild it"
        )
    if not any(destination.iterdir()):
        destination.rmdir()
        os.replace(staging, destination)
        return

    backup = Path(tempfile.mkdtemp(prefix=f".{destination.name}.backup-", dir=destination.parent))
    backup.rmdir()
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    try:
        shutil.rmtree(backup)
    except OSError as exc:
        warnings.warn(
            f"could not remove replaced motion-store backup {backup}: {exc}", stacklevel=2
        )


def _normalized_npz_field_specs(path: Path) -> dict[str, MotionFieldSpec]:
    import numpy as np

    specs: dict[str, MotionFieldSpec] = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            if name == "fps":
                continue
            array = archive[name]
            specs[name] = MotionFieldSpec(
                name=name,
                shape=("num_frames", *tuple(int(size) for size in array.shape[1:])),
                dtype=str(array.dtype),
            )
    return specs


def materialize_paired_sonic_motion(
    robot_root: str | Path,
    smpl_root: str | Path,
    output_dir: str | Path,
    *,
    fps: int,
    joint_order: Sequence[str],
    body_order: Sequence[str],
    source_fps: int | float | None = None,
    quaternion_order: str = "wxyz",
    fk_model_path: str | Path | None = None,
    derive_velocities: bool = True,
    smpl_y_up: bool = False,
    allow_unmatched: bool = False,
    overwrite: bool = False,
    compressed: bool = True,
) -> MotionMaterializationReport:
    """Stream a basename-paired robot/SMPL corpus into an atomic motion store."""

    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise MotionManifestError("fps must be a positive integer")
    joint_names = _order(list(joint_order), "joint_order")
    body_names = _order(list(body_order), "body_order")
    robot_index = _index_paired_source_root(robot_root, label="robot")
    smpl_index = _index_paired_source_root(smpl_root, label="SMPL")
    missing_smpl = sorted(set(robot_index).difference(smpl_index))
    missing_robot = sorted(set(smpl_index).difference(robot_index))
    if missing_smpl or missing_robot:
        summary = (
            f"missing SMPL=[{_summarize_keys(missing_smpl)}], "
            f"missing robot=[{_summarize_keys(missing_robot)}]"
        )
        if not allow_unmatched:
            raise MotionManifestError(f"unmatched paired motion keys: {summary}")
        warnings.warn(f"skipping unmatched paired motion keys: {summary}", stacklevel=2)
    pair_keys = sorted(set(robot_index).intersection(smpl_index))
    if not pair_keys:
        raise MotionManifestError("robot and SMPL source roots have no matched basenames")

    raw_destination = Path(output_dir).expanduser().absolute()
    if raw_destination.is_symlink():
        raise MotionManifestError(
            f"materialization output must not be a symbolic link: {raw_destination}"
        )
    destination = raw_destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (destination.is_symlink() or not destination.is_dir()):
        raise MotionManifestError(
            f"materialization output is not a regular directory: {destination}"
        )
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise MotionManifestError(
            f"materialization output directory is not empty: {destination}; "
            "pass overwrite=True to rebuild it"
        )

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        clips_dir = staging / "clips"
        clips_dir.mkdir()
        field_specs: dict[str, MotionFieldSpec] | None = None
        clips: list[MotionClip] = []
        total_frames = 0
        total_bytes = 0
        for index, pair_key in enumerate(pair_keys):
            merged, pair_fps = _merge_paired_motion_source(
                robot_index[pair_key],
                smpl_index[pair_key],
                pair_key=pair_key,
                source_fps=source_fps,
                target_fps=fps,
                joint_order=joint_names,
                body_order=body_names,
                quaternion_order=quaternion_order,
                fk_model_path=fk_model_path,
                derive_velocities=derive_velocities,
                smpl_y_up=smpl_y_up,
            )
            safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", pair_key).strip("._") or "clip"
            target = clips_dir / f"{index:06d}_{safe_key}.npz"
            report = convert_sonic_motion(
                merged,
                target,
                source_fps=pair_fps,
                target_fps=fps,
                joint_order=joint_names,
                body_order=body_names,
                # ``merged`` is already canonical WXYZ with body poses
                # materialized; the second pass only validates/writes it.
                quaternion_order="wxyz",
                derive_velocities=derive_velocities,
                smpl_y_up=smpl_y_up,
                compressed=compressed,
            )
            current_specs = _normalized_npz_field_specs(target)
            if field_specs is None:
                field_specs = current_specs
            elif current_specs != field_specs:
                raise MotionManifestError(
                    f"pair {pair_key!r} normalized fields/shapes/dtypes disagree with first pair"
                )
            clips.append(
                MotionClip(
                    id=pair_key,
                    path=str(target.relative_to(staging)),
                    checksum=report.checksum,
                    fps=report.fps,
                    num_frames=report.num_frames,
                    joint_order=joint_names,
                    body_order=body_names,
                )
            )
            total_frames += report.num_frames
            total_bytes += target.stat().st_size
            # Do not retain one pair's decoded arrays while opening the next
            # pair. The manifest accumulator above contains metadata only.
            del merged, pair_fps, report, current_specs

        assert field_specs is not None
        manifest = MotionManifest(
            version=MANIFEST_VERSION,
            schema=MANIFEST_SCHEMA,
            joint_order=joint_names,
            body_order=body_names,
            fields=tuple(field_specs.values()),
            clips=tuple(clips),
            metadata={"materializer": "unilab.sonic_motion.paired", "source_count": len(clips)},
            manifest_path=staging / "manifest.json",
        )
        staged_manifest = staging / "manifest.json"
        staged_manifest.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        preflight_motion_manifest(
            staged_manifest,
            verify_checksums=True,
            verify_shapes=True,
            expected_joint_order=joint_names,
            expected_body_order=body_names,
        )
        _publish_staged_motion_store(staging, destination, overwrite=overwrite)
        return MotionMaterializationReport(
            manifest_path=destination / "manifest.json",
            clip_count=len(clips),
            total_frames=total_frames,
            total_bytes=total_bytes,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)


materialize_motion_manifest = materialize_motion_store


read_motion_manifest = load_motion_manifest
resolve_clip_path = resolve_manifest_clip_path


__all__ = [
    "MANIFEST_SCHEMA",
    "MANIFEST_VERSION",
    "MotionClip",
    "MotionField",
    "MotionFieldSpec",
    "MotionManifest",
    "MotionManifestError",
    "MotionConversionReport",
    "MotionMaterializationReport",
    "convert_motion_clip",
    "convert_sonic_motion",
    "load_motion_manifest",
    "materialize_motion_manifest",
    "materialize_paired_sonic_motion",
    "materialize_motion_store",
    "normalize_motion_clip",
    "normalize_sonic_motion",
    "preflight_motion_manifest",
    "read_motion_manifest",
    "resolve_clip_path",
    "resolve_manifest_clip_path",
    "sha256_file",
    "validate_motion_manifest",
]
