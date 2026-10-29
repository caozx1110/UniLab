"""Finite-state diagnostics for the tensor-native environment runtime.

The guard's detection path stays in Torch: it never converts observations,
rewards, controls, or physics snapshots to host arrays.  Host conversion and
artifact writing happen only in :meth:`NanGuard.dump`, which is an
explicit abnormal-diagnostic boundary.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NanGuardCfg:
    """Pickleable configuration for the tensor finite-state guard."""

    enabled: bool = False
    buffer_size: int = 100
    max_envs_to_dump: int = 5
    output_dir: str | None = None


def _bad_row_mask(value: torch.Tensor, num_envs: int) -> torch.Tensor:
    """Return a one-D boolean mask without introducing a host synchronization."""
    if value.ndim == 0 or value.shape[0] != num_envs:
        raise ValueError(
            f"tensor diagnostics leading dimension must be {num_envs}, got {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise TypeError("tensor diagnostics finite checks require floating-point tensors")
    return ~torch.isfinite(value).reshape(num_envs, -1).any(dim=1)


def _snapshot(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", copy=True)


class NanGuard:
    """Detect bad tensor rows on-device and export dumps at an explicit host boundary."""

    def __init__(
        self,
        cfg: NanGuardCfg,
        num_envs: int,
        supports_state_playback: bool,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("tensor diagnostics require a positive environment count")
        if cfg.buffer_size <= 0:
            raise ValueError("tensor diagnostics buffer_size must be positive")
        if cfg.max_envs_to_dump < 0:
            raise ValueError("tensor diagnostics max_envs_to_dump must be non-negative")
        self._cfg = cfg
        self._num_envs = num_envs
        self._supports_state_playback = supports_state_playback
        self._buffer: list[torch.Tensor] = []
        self._buffer_idx = 0
        self._buffer_full = False
        self._dumped = False
        self.deferred_verdict = None

    def _validate_finite_float(self, value: torch.Tensor, label: str) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label} must be a torch.Tensor, got {type(value).__name__}")
        if not value.is_floating_point():
            raise TypeError(f"{label} must be a floating-point tensor, got {value.dtype}")

    def capture(self, physics_state: torch.Tensor | None) -> None:
        """Retain a detached device tensor snapshot for a later abnormal dump."""
        if physics_state is None:
            return
        self._validate_finite_float(physics_state, "physics_state")
        if physics_state.ndim < 2 or physics_state.shape[0] != self._num_envs:
            raise ValueError(
                "physics_state must have shape "
                f"({self._num_envs}, ...), got {tuple(physics_state.shape)}"
            )
        snapshot = physics_state.detach().clone()
        if not self._buffer_full and len(self._buffer) < self._cfg.buffer_size:
            self._buffer.append(snapshot)
        else:
            self._buffer_full = True
            self._buffer[self._buffer_idx] = snapshot
        self._buffer_idx = (self._buffer_idx + 1) % self._cfg.buffer_size

    def check(
        self,
        obs: Mapping[str, torch.Tensor],
        reward: torch.Tensor,
        step: int = 0,
    ) -> torch.Tensor | None:
        """Return device-resident bad-row indices, or ``None`` when every row is finite."""
        if not self._cfg.enabled:
            return None
        if not isinstance(obs, Mapping):
            raise TypeError(f"observations must be a mapping, got {type(obs).__name__}")
        bad_mask = torch.zeros((self._num_envs,), dtype=torch.bool, device=reward.device if self.deferred_verdict is not None else "cpu")
        for name, value in obs.items():
            self._validate_finite_float(value, f"obs[{name!r}]")
            bad_mask |= _bad_row_mask(value, self._num_envs).to(bad_mask.device)
        self._validate_finite_float(reward, "reward")
        reward_mask = _bad_row_mask(reward, self._num_envs)
        bad_mask |= reward_mask.to(bad_mask.device)
        if self.deferred_verdict is not None:
            self.deferred_verdict(~bad_mask.any())
            return None
        if not bool(bad_mask.any()):
            return None
        nan_ids = (
            torch.nonzero(bad_mask, as_tuple=False)
            .flatten()
            .to(device=reward.device, dtype=torch.int64)
        )
        logger.warning(
            "NanGuard: NaN/Inf detected in obs/reward at step %d (envs=%d, sample_ids=%s)",
            step,
            nan_ids.numel(),
            nan_ids[: min(5, nan_ids.numel())].detach().cpu().tolist(),
        )
        return nan_ids

    def check_ctrl(self, ctrl: torch.Tensor, step: int = 0) -> torch.Tensor | None:
        """Return device-resident bad control-row indices, or ``None`` when finite."""
        if not self._cfg.enabled:
            return None
        self._validate_finite_float(ctrl, "ctrl")
        bad_mask = _bad_row_mask(ctrl, self._num_envs)
        if self.deferred_verdict is not None:
            self.deferred_verdict(~bad_mask.any())
            return None
        if not bool(bad_mask.any()):
            return None
        nan_ids = (
            torch.nonzero(bad_mask, as_tuple=False)
            .flatten()
            .to(device=ctrl.device, dtype=torch.int64)
        )
        logger.warning(
            "NanGuard: NaN/Inf detected in ctrl at step %d (envs=%d, sample_ids=%s)",
            step,
            nan_ids.numel(),
            nan_ids[: min(5, nan_ids.numel())].detach().cpu().tolist(),
        )
        return nan_ids

    def dump(
        self,
        nan_env_ids: torch.Tensor,
        model_file: str,
        step: int,
    ) -> str | None:
        """Export one abnormal-state artifact at an explicit host boundary."""
        if self._dumped:
            return None
        self._dumped = True
        if not isinstance(nan_env_ids, torch.Tensor):
            raise TypeError(f"nan_env_ids must be a torch.Tensor, got {type(nan_env_ids).__name__}")
        if nan_env_ids.ndim != 1:
            raise ValueError("nan_env_ids must be one-dimensional")
        host_nan_ids = _snapshot(nan_env_ids).to(dtype=torch.int64)

        output_dir = Path(self._cfg.output_dir or "/tmp/unilab/nan_dumps")
        output_dir.mkdir(parents=True, exist_ok=True)
        dump_env_ids = host_nan_ids[: self._cfg.max_envs_to_dump]

        if self._buffer_full:
            ordered = self._buffer[self._buffer_idx :] + self._buffer[: self._buffer_idx]
        else:
            ordered = list(self._buffer)
        states = (
            torch.stack(ordered, dim=0)
            if ordered
            else torch.empty((0, self._num_envs, 0), dtype=torch.float32)
        )
        host_states = _snapshot(states).numpy()
        if host_states.ndim >= 3 and dump_env_ids.numel() > 0:
            host_states = host_states[:, dump_env_ids.numpy()]

        metadata = {
            "num_envs_total": self._num_envs,
            "nan_env_ids": host_nan_ids.numpy(),
            "dumped_env_ids": dump_env_ids.numpy(),
            "buffer_size": self._cfg.buffer_size,
            "buffer_len": len(ordered),
            "detection_step": step,
            "timestamp": time.time(),
            "model_file": model_file,
            "supports_state_playback": self._supports_state_playback,
        }

        ts = time.strftime("%Y%m%d_%H%M%S")
        dump_name = f"nan_dump_{ts}_step{step}"
        npz_path = output_dir / f"{dump_name}.npz"
        np.savez(
            str(npz_path),
            states=host_states,
            **{f"meta_{key}": value for key, value in metadata.items()},
        )

        if model_file and Path(model_file).is_file():
            model_dst = output_dir / f"{dump_name}_model{Path(model_file).suffix}"
            shutil.copy2(model_file, model_dst)

        latest_link = output_dir / "nan_dump_latest.npz"
        latest_link.unlink(missing_ok=True)
        try:
            latest_link.symlink_to(npz_path.name)
        except OSError:
            pass

        logger.info(
            "NanGuard: dump written to %s (step=%d, envs=%d)",
            npz_path,
            step,
            host_nan_ids.numel(),
        )
        return str(npz_path)
