"""SONIC original-release owner for the UniLab G1 MuJoCo environment.

The regular :class:`G1MotionTrackingEnv` intentionally keeps its historical
flat actor/critic contract.  SONIC needs a separate owner because its policy
history, tokenizer modalities and future-reference windows are part of the
network contract.  This module reuses the shared tracking/reward engine and
only owns the SONIC observation and action-order boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import gymnasium as gym
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.g1.base import NoiseConfig
from unilab.envs.motion_tracking.common.config import MotionTrackingCfg
from unilab.envs.motion_tracking.common.tracking import MotionTrackingEnv
from unilab.utils.geometry import np_write_relative_anchor_transform_pos_rot6d
from unilab.utils.rotation import (
    np_matrix_first_two_cols_from_quat,
    np_quat_apply_batched,
    np_quat_apply_inverse,
    np_quat_conjugate_batched,
    np_quat_error_magnitude_squared_batched,
    np_quat_mul_batched,
)

from ..common.rewards import (
    anti_shake_ang_vel,
    joint_acc_l2,
    tracking_vr_5point_local,
)

if TYPE_CHECKING:
    from unilab.training.sonic_store import SonicMotionStore


SONIC_ACTOR_OBS_DIM = 930
SONIC_CRITIC_OBS_DIM = 1645
SONIC_TOKENIZER_OBS_DIM = 1761
SONIC_RELEASE_REVISION = "c374bae5b9039cd0ee71377e654d11ce1bc69e1d"
SONIC_RELEASE_OBSERVATION_PROFILE = "unitoken_all_noz"
SONIC_TERMINATION_DOWN_THRESHOLD = 0.75
SONIC_TERMINATION_ROOT_HEIGHT_THRESHOLD = 0.5
SONIC_TERMINATION_FOOT_POS_THRESHOLD = 0.2
SONIC_TERMINATION_ROOT_HEIGHT_EMA_ALPHA = 0.1


class _SonicReleaseMotionSampler:
    """Release-local adaptive curriculum over clip-local fixed-size bins.

    The shared :class:`MotionSampler` indexes one concatenated corpus, which
    gives longer clips more initial sampling mass and cannot represent the
    release curriculum's per-motion bins.  SONIC owns this small replacement
    rather than changing the common tracking contract.  All clip/bin metadata
    is materialized here, during environment construction; reset and step only
    operate on those arrays and never inspect the manifest or an asset.
    """

    def __init__(
        self,
        motion_loader: Any,
        *,
        num_envs: int,
        bin_size: int,
        init_num_failures: float,
        uniform_sampling_rate: float,
        failure_rate_max_over_mean: float,
        pre_failure_sample_window: int,
        sequence_length_agnostic: bool,
        probability_refresh_interval_steps: int,
        active_motion_pool_size: int = 1024,
        motion_resample_frequency: int = 250,
        freeze_frame_aug: bool = False,
        freeze_frame_prob: float = 0.1,
    ) -> None:
        if bin_size < 1:
            raise ValueError(f"SONIC adaptive bin_size must be positive, got {bin_size}")
        if init_num_failures <= 0.0:
            raise ValueError(
                f"SONIC adaptive init_num_failures must be positive, got {init_num_failures}"
            )
        if not 0.0 <= uniform_sampling_rate <= 1.0:
            raise ValueError(
                "SONIC adaptive uniform_sampling_rate must be in [0, 1], "
                f"got {uniform_sampling_rate}"
            )
        if failure_rate_max_over_mean <= 0.0:
            raise ValueError(
                "SONIC adaptive failure_rate_max_over_mean must be positive, "
                f"got {failure_rate_max_over_mean}"
            )
        if pre_failure_sample_window < 0:
            raise ValueError(
                "SONIC adaptive pre_failure_sample_window must be non-negative, "
                f"got {pre_failure_sample_window}"
            )
        if probability_refresh_interval_steps < 1:
            raise ValueError(
                "SONIC adaptive probability_refresh_interval_steps must be positive, "
                f"got {probability_refresh_interval_steps}"
            )
        if active_motion_pool_size < 1:
            raise ValueError(
                f"SONIC active_motion_pool_size must be positive, got {active_motion_pool_size}"
            )
        if motion_resample_frequency < 1:
            raise ValueError(
                f"SONIC motion_resample_frequency must be positive, got {motion_resample_frequency}"
            )
        if not 0.0 <= freeze_frame_prob <= 1.0:
            raise ValueError(f"SONIC freeze_frame_prob must be in [0, 1], got {freeze_frame_prob}")

        self.motion_loader = motion_loader
        self.mode = "adaptive"
        self.num_envs = int(num_envs)
        self.bin_size = int(bin_size)
        self.uniform_sampling_rate = float(uniform_sampling_rate)
        self.failure_rate_max_over_mean = float(failure_rate_max_over_mean)
        self.pre_failure_sample_window = int(pre_failure_sample_window)
        self.sequence_length_agnostic = bool(sequence_length_agnostic)
        self.active_motion_pool_size = int(active_motion_pool_size)
        self.motion_resample_frequency = int(motion_resample_frequency)
        self.freeze_frame_aug = bool(freeze_frame_aug)
        self.freeze_frame_prob = float(freeze_frame_prob)
        # The release recomputes this distribution once at the end of each
        # PPO collection.  This owner has no runner hook by design, so count
        # per-step outcome updates instead.  The owner YAML pins this to the
        # 24-step SONIC rollout horizon.  Crucially, raw exposure/failure
        # statistics still update on every step; only their sampled view is
        # batched.
        self.probability_refresh_interval_steps = int(probability_refresh_interval_steps)

        clip_lengths = np.asarray(motion_loader.clip_lengths, dtype=np.int32)
        clip_offsets = np.asarray(motion_loader.clip_offsets, dtype=np.int32)
        if clip_lengths.ndim != 1 or clip_lengths.size == 0 or np.any(clip_lengths <= 0):
            raise ValueError("SONIC motion loader must expose non-empty positive clip_lengths")
        if clip_offsets.shape != clip_lengths.shape:
            raise ValueError("SONIC motion loader clip_offsets must match clip_lengths")
        self._clip_lengths = clip_lengths
        self._clip_offsets = clip_offsets
        self._clip_end_frames = np.asarray(motion_loader.clip_end_frames, dtype=np.int32)
        self._bins_per_clip = (clip_lengths + self.bin_size - 1) // self.bin_size
        self._bin_offsets = np.empty_like(self._bins_per_clip)
        self._bin_offsets[0] = 0
        if self._bin_offsets.size > 1:
            np.cumsum(self._bins_per_clip[:-1], out=self._bin_offsets[1:])
        self._bin_clip_indices = np.repeat(
            np.arange(clip_lengths.size, dtype=np.int32), self._bins_per_clip
        )
        self._bin_local_starts = np.concatenate(
            tuple(
                np.arange(0, int(length), self.bin_size, dtype=np.int32) for length in clip_lengths
            )
        )
        self._bin_local_ends = np.minimum(
            self._bin_local_starts + self.bin_size,
            clip_lengths[self._bin_clip_indices],
        )
        self._bin_lengths = (self._bin_local_ends - self._bin_local_starts).astype(
            np.float64, copy=False
        )
        self.bin_count = int(self._bin_lengths.size)
        if self.bin_count < 1:  # defensive: positive clips always create a bin
            raise ValueError("SONIC adaptive sampler requires at least one bin")

        # This exactly follows the release's bin-length correction, then
        # divides by the number of bins in that clip when sequence lengths are
        # deliberately made agnostic.
        self._bin_weights = self._bin_lengths / self._bin_lengths.mean()
        if self.sequence_length_agnostic:
            self._bin_weights /= self._bins_per_clip[self._bin_clip_indices]

        self.bin_episode_count = np.full(self.bin_count, init_num_failures, dtype=np.float64)
        self.bin_failed_count = np.full(self.bin_count, init_num_failures, dtype=np.float64)
        self._sampling_prob = np.empty(self.bin_count, dtype=np.float64)
        # ``np.random.choice(..., p=...)`` materializes a cumulative
        # distribution on every reset.  Full-corpus training has O(10^5)
        # bins, so keep that cumulative distribution alongside probabilities
        # and search it directly on the reset hot path.
        self._sampling_cdf = np.empty(self.bin_count, dtype=np.float64)
        self._motion_sampling_prob = np.empty(clip_lengths.size, dtype=np.float64)
        self._steps_since_probability_refresh = 0

        self.current_frames = np.zeros(self.num_envs, dtype=np.int32)
        # The release freezes the materialized reference after a selected
        # frame, while clip progression and boundary resampling continue.
        self._timeline_frames = self.current_frames.copy()
        # ``_clip_freeze_frames`` remains the clip-keyed compatibility view for
        # callers without an environment id.  The actual release semantics are
        # slot-keyed below: a duplicated clip in one active pool can receive
        # independent cutoff draws.
        self._clip_freeze_frames = self._clip_end_frames.copy()
        self.current_clip_indices = np.zeros(self.num_envs, dtype=np.int32)
        self.current_clip_end_frames = np.full(
            self.num_envs, self._clip_end_frames[0], dtype=np.int32
        )
        self.current_active_slots = np.zeros(self.num_envs, dtype=np.int32)
        self._done_mask = np.zeros(self.num_envs, dtype=bool)

        # Keep the public metric names exposed by MotionSampler for existing
        # logging consumers, even though the bins are intentionally private to
        # this SONIC owner.
        self.sampling_entropy = 1.0
        self.sampling_top1_prob = 1.0
        self.sampling_top1_bin = 0.0
        self._all_motions_active = self.active_motion_pool_size >= len(self._clip_lengths)
        self.active_clip_indices = np.empty(0, dtype=np.int32)
        self.active_slot_freeze_frames = np.empty(0, dtype=np.int32)
        self._active_bin_ids = np.empty(0, dtype=np.int32)
        self._active_bin_slots = np.empty(0, dtype=np.int32)
        self._refresh_global_motion_sampling_probabilities()
        self._load_active_motion_pool()
        self._refresh_sampling_probabilities()

    def _bin_sampling_probability(self, bin_ids: np.ndarray) -> np.ndarray:
        """Compute the release distribution for one view of global bins.

        The global distribution chooses the next active pool.  The active view
        then recomputes the same expression over only that pool's bin slots.
        Keeping these two normalizations separate is material: the release's
        uniform mixture is normalized over active bins, not over the corpus.
        """

        bin_ids = np.asarray(bin_ids, dtype=np.intp)
        if bin_ids.ndim != 1 or bin_ids.size == 0:
            raise ValueError("SONIC adaptive sampling requires a non-empty bin view")
        failure_rate = self.bin_failed_count[bin_ids] / self.bin_episode_count[bin_ids]
        upper_bound = float(failure_rate.mean() * self.failure_rate_max_over_mean)
        capped_failure_rate = np.clip(failure_rate, 0.0, upper_bound)
        failure_mass = capped_failure_rate.sum()
        if failure_mass <= 0.0 or not np.isfinite(failure_mass):
            failure_prob = np.full(bin_ids.size, 1.0 / bin_ids.size, dtype=np.float64)
        else:
            failure_prob = capped_failure_rate / failure_mass
        probability = (
            (1.0 - self.uniform_sampling_rate) * failure_prob
            + self.uniform_sampling_rate / bin_ids.size
        ) * self._bin_weights[bin_ids]
        probability_mass = probability.sum()
        if probability_mass <= 0.0 or not np.isfinite(probability_mass):
            probability.fill(1.0 / bin_ids.size)
        else:
            probability /= probability_mass
        return probability

    def _refresh_global_motion_sampling_probabilities(self) -> None:
        """Aggregate full-corpus bin probabilities into motion probabilities."""

        global_bin_ids = np.arange(self.bin_count, dtype=np.int32)
        global_probability = self._bin_sampling_probability(global_bin_ids)
        motion_probability = np.bincount(
            self._bin_clip_indices,
            weights=global_probability,
            minlength=len(self._clip_lengths),
        ).astype(np.float64, copy=False)
        probability_mass = motion_probability.sum()
        if probability_mass <= 0.0 or not np.isfinite(probability_mass):
            motion_probability.fill(1.0 / motion_probability.size)
        else:
            motion_probability /= probability_mass
        self._motion_sampling_prob[...] = motion_probability

    def _load_active_motion_pool(self) -> bool:
        """Materialize one release active-pool view without touching payloads.

        Source SONIC loads the full corpus once when it fits the requested
        limit, then future callback invocations return ``False`` and do not
        reset.  The equivalent here is a sequential, duplicate-free full pool
        for that first construction and a no-op thereafter.  Otherwise slots
        are sampled with replacement, matching ``torch.multinomial``.
        """

        if self._all_motions_active and self.active_clip_indices.size:
            return False
        if self._all_motions_active:
            active_clip_indices = np.arange(len(self._clip_lengths), dtype=np.int32)
        else:
            motion_cdf = np.cumsum(self._motion_sampling_prob)
            motion_cdf[-1] = 1.0
            active_clip_indices = np.searchsorted(
                motion_cdf,
                np.random.random(self.active_motion_pool_size),
                side="right",
            ).astype(np.int32, copy=False)
            np.minimum(active_clip_indices, len(self._clip_lengths) - 1, out=active_clip_indices)

        slot_counts = self._bins_per_clip[active_clip_indices]
        active_bin_ids = np.concatenate(
            tuple(
                np.arange(
                    self._bin_offsets[clip_index],
                    self._bin_offsets[clip_index] + self._bins_per_clip[clip_index],
                    dtype=np.int32,
                )
                for clip_index in active_clip_indices
            )
        )
        active_bin_slots = np.repeat(
            np.arange(active_clip_indices.size, dtype=np.int32), slot_counts
        )
        if active_bin_ids.size != active_bin_slots.size:
            raise RuntimeError("SONIC active-pool bin/slot construction disagrees")

        slot_freeze_frames = self._clip_end_frames[active_clip_indices].copy()
        if self.freeze_frame_aug:
            for slot, clip_index in enumerate(active_clip_indices):
                if np.random.random() < self.freeze_frame_prob:
                    slot_freeze_frames[slot] = int(self._clip_offsets[clip_index]) + int(
                        np.random.randint(0, int(self._clip_lengths[clip_index]))
                    )

        self.active_clip_indices = active_clip_indices
        self.active_slot_freeze_frames = slot_freeze_frames
        self._active_bin_ids = active_bin_ids
        self._active_bin_slots = active_bin_slots
        # Compatibility path for callers that lack the active slot.  All
        # owner observation call-sites pass ``env_ids`` and therefore use the
        # slot-accurate cutoff above.
        self._clip_freeze_frames[...] = self._clip_end_frames
        self._clip_freeze_frames[active_clip_indices] = slot_freeze_frames
        return True

    def maybe_resample_active_motion_pool(self, completed_global_steps: int) -> bool:
        """Reload active slots on the release's PPO global-step cadence.

        The native SONIC runner owns invocation after a completed PPO update;
        generic runners and the shared environment lifecycle are untouched.
        A true result tells that runner to reset all environments before its
        next collection, matching the source wrapper's ``reset_all()``.
        """

        if (
            isinstance(completed_global_steps, bool)
            or not isinstance(completed_global_steps, int)
            or completed_global_steps < 0
        ):
            raise ValueError("SONIC completed_global_steps must be a non-negative integer")
        if completed_global_steps == 0 or (completed_global_steps % self.motion_resample_frequency):
            return False
        if self._all_motions_active:
            return False
        self._refresh_global_motion_sampling_probabilities()
        changed = self._load_active_motion_pool()
        if changed:
            self._refresh_sampling_probabilities()
        return changed

    def reload_active_motion_pool_after_restore(self) -> bool:
        """Rebuild the source-style active view after counters are restored.

        The upstream wrapper restores only global adaptive counters, refreshes
        their probability view, and invokes its motion reload path.  A corpus
        that was entirely loaded remains loaded; otherwise reload recreates
        slots and their per-load freeze decisions from the restored prior.
        """

        self._refresh_global_motion_sampling_probabilities()
        if self._all_motions_active:
            self._refresh_sampling_probabilities()
            return False
        changed = self._load_active_motion_pool()
        if not changed:  # pragma: no cover - defensive against future pool modes.
            raise RuntimeError("SONIC active-pool restore unexpectedly did not reload")
        self._refresh_sampling_probabilities()
        return True

    def _refresh_sampling_probabilities(self) -> None:
        """Refresh only the current active-bin view used by reset sampling."""

        self._sampling_prob = self._bin_sampling_probability(self._active_bin_ids)
        self._sampling_cdf = np.cumsum(self._sampling_prob)
        # Avoid a floating-point tail below one turning the largest valid
        # uniform draw into an out-of-bounds index.
        self._sampling_cdf[-1] = 1.0
        active_bin_count = len(self._active_bin_ids)
        entropy = -np.sum(self._sampling_prob * np.log(self._sampling_prob + 1.0e-12))
        self.sampling_entropy = (
            float(entropy / np.log(active_bin_count)) if active_bin_count > 1 else 1.0
        )
        top_position = int(np.argmax(self._sampling_prob))
        self.sampling_top1_prob = float(self._sampling_prob[top_position])
        self.sampling_top1_bin = float(self._active_bin_ids[top_position]) / self.bin_count
        self._steps_since_probability_refresh = 0

    def _slots_for_clips(self, clip_indices: np.ndarray) -> np.ndarray:
        """Resolve compatibility calls to the first loaded slot of each clip."""

        slots = np.empty(len(clip_indices), dtype=np.int32)
        for position, clip_index in enumerate(clip_indices):
            matches = np.flatnonzero(self.active_clip_indices == clip_index)
            if matches.size == 0:
                raise ValueError("SONIC sampled a clip outside its active motion pool")
            slots[position] = matches[0]
        return slots

    def _set_sampled_frames(
        self,
        env_ids: np.ndarray,
        frames: np.ndarray,
        active_slots: np.ndarray | None = None,
    ) -> None:
        frames = np.asarray(frames, dtype=np.int32)
        if frames.shape != (len(env_ids),):
            raise ValueError("SONIC sampled frames must have one row per environment id")
        self._timeline_frames[env_ids] = frames
        clip_indices = self.motion_loader.get_clip_indices(frames)
        if active_slots is None:
            active_slots = self._slots_for_clips(clip_indices)
        active_slots = np.asarray(active_slots, dtype=np.int32)
        if active_slots.shape != (len(env_ids),) or np.any(
            (active_slots < 0) | (active_slots >= len(self.active_clip_indices))
        ):
            raise ValueError("SONIC active slots must match sampled environment ids")
        if not np.array_equal(self.active_clip_indices[active_slots], clip_indices):
            raise ValueError("SONIC active slot does not own its sampled clip")
        self.current_active_slots[env_ids] = active_slots
        self.current_clip_indices[env_ids] = clip_indices
        self.current_clip_end_frames[env_ids] = self._clip_end_frames[clip_indices]
        self.current_frames[env_ids] = np.minimum(
            frames, self.active_slot_freeze_frames[active_slots]
        )

    @property
    def timeline_frames(self) -> np.ndarray:
        """Raw clip progression, before any reference freeze clamp."""

        return self._timeline_frames

    def effective_reference_frames(self, frames: np.ndarray) -> np.ndarray:
        """Map absolute raw frames through the compatibility clip cutoff."""

        frames = np.asarray(frames, dtype=np.int64)
        clip_indices = np.searchsorted(self._clip_offsets, frames, side="right") - 1
        clip_indices = np.clip(clip_indices, 0, len(self._clip_lengths) - 1)
        return np.minimum(frames, self._clip_freeze_frames[clip_indices]).astype(
            np.int32, copy=False
        )

    def clamp_reference_indices(
        self, indices: np.ndarray, env_ids: np.ndarray | None = None
    ) -> np.ndarray:
        """Clamp arbitrary reference matrices to active-slot freeze cutoffs."""

        values = np.asarray(indices, dtype=np.int64)
        if env_ids is None:
            return self.effective_reference_frames(values)
        env_ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
        if values.ndim < 1 or values.shape[0] != len(env_ids):
            raise ValueError("SONIC reference rows must match env_ids for slot-level freeze")
        if np.any((env_ids < 0) | (env_ids >= self.num_envs)):
            raise IndexError("SONIC reference env id out of bounds")
        cutoffs = self.active_slot_freeze_frames[self.current_active_slots[env_ids]]
        cutoff_shape = (len(env_ids),) + (1,) * (values.ndim - 1)
        return np.minimum(values, cutoffs.reshape(cutoff_shape)).astype(np.int32, copy=False)

    def sample_frames(self, env_ids: np.ndarray) -> np.ndarray:
        env_ids = np.asarray(env_ids, dtype=np.intp)
        if env_ids.ndim != 1:
            raise ValueError("SONIC adaptive sampler env_ids must be one-dimensional")
        if not env_ids.size:
            return np.empty(0, dtype=np.int32)
        if np.any((env_ids < 0) | (env_ids >= self.num_envs)):
            raise IndexError("SONIC adaptive sampler env id out of bounds")
        sampled_positions = np.searchsorted(
            self._sampling_cdf, np.random.random(env_ids.size), side="right"
        )
        np.minimum(sampled_positions, len(self._active_bin_ids) - 1, out=sampled_positions)
        sampled_bins = self._active_bin_ids[sampled_positions]
        starts = self._bin_local_starts[sampled_bins]
        spans = self._bin_local_ends[sampled_bins] - starts
        local_frames = starts + (np.random.random(env_ids.size) * spans).astype(np.int32)
        if self.pre_failure_sample_window:
            local_frames -= np.random.randint(
                self.pre_failure_sample_window, size=env_ids.size, dtype=np.int32
            )
            np.maximum(local_frames, 0, out=local_frames)
        frames = self._clip_offsets[self._bin_clip_indices[sampled_bins]] + local_frames
        frames = np.asarray(frames, dtype=np.int32)
        self._set_sampled_frames(env_ids, frames, self._active_bin_slots[sampled_positions])
        return self.current_frames[env_ids]

    def update_failure_stats(
        self, terminated: np.ndarray, current_frames: np.ndarray | None = None
    ) -> None:
        """Record outcomes by ``(clip id, clip-local bin)``.

        The release updates bin exposure on *every* simulation step, normalized
        by that bin's frame count; failures receive one additional unnormalized
        count.  Clip boundaries are resampled by :meth:`step` and do not enter
        this method as terminations.
        """
        done = np.asarray(terminated, dtype=bool)
        if done.shape != (self.num_envs,):
            raise ValueError(
                "SONIC adaptive sampler terminated mask must have shape "
                f"({self.num_envs},), got {done.shape}"
            )
        if current_frames is None:
            frames = self._timeline_frames
            clip_indices = self.current_clip_indices
        else:
            frames = np.asarray(current_frames, dtype=np.int32)
            if frames.shape != (self.num_envs,):
                raise ValueError(
                    "SONIC adaptive sampler current_frames must have shape "
                    f"({self.num_envs},), got {frames.shape}"
                )
            clip_indices = self.motion_loader.get_clip_indices(frames)

        local_frames = np.clip(
            frames - self._clip_offsets[clip_indices],
            0,
            self._clip_lengths[clip_indices] - 1,
        )
        local_bins = np.minimum(
            local_frames // self.bin_size, self._bins_per_clip[clip_indices] - 1
        )
        bin_indices = self._bin_offsets[clip_indices] + local_bins
        np.add.at(self.bin_episode_count, bin_indices, 1.0 / self._bin_lengths[bin_indices])
        if np.any(done):
            np.add.at(self.bin_failed_count, bin_indices[done], 1.0)
        self._steps_since_probability_refresh += 1
        if self._steps_since_probability_refresh >= self.probability_refresh_interval_steps:
            self._refresh_sampling_probabilities()

    def step(self) -> np.ndarray:
        self._timeline_frames += 1
        np.greater(self._timeline_frames, self.current_clip_end_frames, out=self._done_mask)
        np.copyto(self.current_frames, self._timeline_frames)
        np.minimum(
            self.current_frames,
            self.active_slot_freeze_frames[self.current_active_slots],
            out=self.current_frames,
        )
        return np.flatnonzero(self._done_mask)

    def get_current_motion(self, out: Any | None = None) -> Any:
        return self.motion_loader.get_motion_at_frame(self.current_frames, out=out)


@dataclass(frozen=True)
class SonicObservationTerm:
    """One immutable term in a flattened SONIC observation group.

    The order below is the resolved release order, not an alphabetical view of
    the term names.  ``shape`` excludes the environment batch dimension and
    ``flat_slice`` makes the policy ABI directly auditable.
    """

    name: str
    shape: tuple[int, ...]
    start: int
    stop: int

    @property
    def width(self) -> int:
        return self.stop - self.start

    @property
    def flat_slice(self) -> slice:
        return slice(self.start, self.stop)


def _sonic_layout(
    entries: Sequence[tuple[str, tuple[int, ...]]],
) -> tuple[SonicObservationTerm, ...]:
    offset = 0
    result: list[SonicObservationTerm] = []
    for name, shape in entries:
        width = int(np.prod(shape, dtype=np.int64))
        result.append(SonicObservationTerm(name, shape, offset, offset + width))
        offset += width
    return tuple(result)


# Provenance: GR00T-WholeBodyControl@SONIC_RELEASE_REVISION and IsaacLab
# v2.3.2 ObservationManager._prepare_terms.  The manager iterates the
# configclass ``__dict__``, so class declaration order wins over Hydra's
# visually listed defaults order.
#   gear_sonic/config/manager_env/observations/{policy/local_dir_hist,
#   critic/privileged_mf_hist,tokenizer/unitoken_all_noz}.yaml and
#   gear_sonic/envs/manager_env/mdp/observations.py
# The 1761-wide tokenizer is the *training* observation group.  The release
# deploy encoder's 1751 input is a different export ABI: one scalar mode plus
# the active 1750-wide union (the two unused command-z terms are omitted).
SONIC_ACTOR_OBSERVATION_TERMS = _sonic_layout(
    (
        ("base_ang_vel", (10, 3)),
        ("joint_pos", (10, 29)),
        ("joint_vel", (10, 29)),
        ("actions", (10, 29)),
        ("gravity_dir", (10, 3)),
    )
)
SONIC_CRITIC_OBSERVATION_TERMS = _sonic_layout(
    (
        ("command_multi_future", (580,)),
        ("motion_anchor_pos_b", (3,)),
        ("motion_anchor_ori_b", (6,)),
        ("body_pos", (14, 3)),
        ("body_ori", (14, 6)),
        ("base_lin_vel", (10, 3)),
        ("base_ang_vel", (10, 3)),
        ("joint_pos", (10, 29)),
        ("joint_vel", (10, 29)),
        ("actions", (10, 29)),
    )
)
SONIC_TOKENIZER_OBSERVATION_TERMS = _sonic_layout(
    (
        ("encoder_index", (3,)),
        ("command_multi_future_nonflat", (10, 58)),
        ("command_z_multi_future_nonflat", (10, 1)),
        ("command_z", (1,)),
        ("motion_anchor_ori_b", (6,)),
        ("motion_anchor_ori_b_mf_nonflat", (10, 6)),
        ("command_multi_future_lower_body", (240,)),
        ("vr_3point_local_target", (9,)),
        ("vr_3point_local_orn_target", (12,)),
        ("smpl_joints_multi_future_local_nonflat", (10, 72)),
        ("smpl_root_ori_b_multi_future", (10, 6)),
        ("joint_pos_multi_future_wrist_for_smpl", (10, 6)),
    )
)


def pack_sonic_observation_terms(
    terms: Mapping[str, np.ndarray], layout: Sequence[SonicObservationTerm]
) -> np.ndarray:
    """Validate and flatten named terms into their immutable release slices."""

    if not layout:
        raise ValueError("SONIC observation layout must not be empty")
    expected_names = tuple(term.name for term in layout)
    if set(terms) != set(expected_names):
        raise ValueError(
            "SONIC observation terms disagree with layout: "
            f"missing={sorted(set(expected_names) - set(terms))}, "
            f"extra={sorted(set(terms) - set(expected_names))}"
        )
    batch_size: int | None = None
    result: np.ndarray | None = None
    for term in layout:
        value = np.asarray(terms[term.name])
        if value.ndim < 1 or tuple(value.shape[1:]) != term.shape:
            raise ValueError(
                f"SONIC term {term.name!r} must have shape (N, {term.shape}), got {value.shape}"
            )
        if batch_size is None:
            batch_size = int(value.shape[0])
            result = np.empty((batch_size, layout[-1].stop), dtype=np.float32)
        elif value.shape[0] != batch_size:
            raise ValueError(f"SONIC term {term.name!r} has a different batch size")
        assert result is not None
        result[:, term.flat_slice] = value.reshape(value.shape[0], -1)
    assert result is not None
    return result


assert SONIC_ACTOR_OBSERVATION_TERMS[-1].stop == SONIC_ACTOR_OBS_DIM
assert SONIC_CRITIC_OBSERVATION_TERMS[-1].stop == SONIC_CRITIC_OBS_DIM
assert SONIC_TOKENIZER_OBSERVATION_TERMS[-1].stop == SONIC_TOKENIZER_OBS_DIM

# Materialized motion and MuJoCo actuator order.  The release policy itself
# uses the interleaved IsaacLab order declared separately below.
SONIC_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

SONIC_POLICY_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
SONIC_MUJOCO_TO_POLICY: tuple[int, ...] = tuple(
    SONIC_JOINT_ORDER.index(name) for name in SONIC_POLICY_JOINT_ORDER
)
SONIC_POLICY_TO_MUJOCO: tuple[int, ...] = tuple(
    SONIC_POLICY_JOINT_ORDER.index(name) for name in SONIC_JOINT_ORDER
)
SONIC_LOWER_BODY_POLICY_INDICES: tuple[int, ...] = tuple(
    SONIC_POLICY_JOINT_ORDER.index(name) for name in SONIC_JOINT_ORDER[:12]
)

SONIC_BODY_ORDER: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

# Upstream selects these directly from its IsaacLab-policy-order future q.
SONIC_WRIST_JOINT_INDICES: tuple[int, ...] = (23, 24, 25, 26, 27, 28)

SONIC_VR_BODY_ORDER: tuple[str, ...] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "torso_link",
)
SONIC_VR_BODY_OFFSETS: tuple[tuple[float, float, float], ...] = (
    (0.18, -0.025, 0.0),
    (0.18, 0.025, 0.0),
    (0.0, 0.0, 0.35),
)


def sonic_action_scale() -> np.ndarray:
    """Return the release model-12 scale in IsaacLab policy order.

    Isaac's implicit actuator uses ``0.25 * effort_limit / stiffness``.  The
    UniLab G1 XML exposes the same position gains, so keeping this calculation
    explicit avoids silently falling back to the historical scalar ``0.25``.
    """

    natural_frequency = 10.0 * 2.0 * np.pi
    stiffness_5020 = 0.003609725 * natural_frequency**2
    stiffness_7520_14 = 0.010177520 * natural_frequency**2
    stiffness_7520_22 = 0.025101925 * natural_frequency**2
    stiffness_4010 = 0.00425 * natural_frequency**2
    values: dict[str, float] = {}
    for name in ("left", "right"):
        # G1 model-12 uses the 7520-22 actuator (139 Nm) for hip pitch.
        # Hip yaw is the 7520-14 / 88 Nm actuator below.  Mixing these two
        # entries changes the policy-to-torque contract by ~56% on hip pitch.
        values[f"{name}_hip_pitch_joint"] = 0.25 * 139.0 / stiffness_7520_22
        values[f"{name}_hip_roll_joint"] = 0.25 * 139.0 / stiffness_7520_22
        values[f"{name}_hip_yaw_joint"] = 0.25 * 88.0 / stiffness_7520_14
        values[f"{name}_knee_joint"] = 0.25 * 139.0 / stiffness_7520_22
        values[f"{name}_ankle_pitch_joint"] = 0.25 * 50.0 / (2.0 * stiffness_5020)
        values[f"{name}_ankle_roll_joint"] = 0.25 * 50.0 / (2.0 * stiffness_5020)
        values[f"{name}_shoulder_pitch_joint"] = 0.25 * 25.0 / stiffness_5020
        values[f"{name}_shoulder_roll_joint"] = 0.25 * 25.0 / stiffness_5020
        values[f"{name}_shoulder_yaw_joint"] = 0.25 * 25.0 / stiffness_5020
        values[f"{name}_elbow_joint"] = 0.25 * 25.0 / stiffness_5020
        values[f"{name}_wrist_roll_joint"] = 0.25 * 25.0 / stiffness_5020
        values[f"{name}_wrist_pitch_joint"] = 0.25 * 5.0 / stiffness_4010
        values[f"{name}_wrist_yaw_joint"] = 0.25 * 5.0 / stiffness_4010
    values["waist_yaw_joint"] = 0.25 * 88.0 / stiffness_7520_14
    values["waist_roll_joint"] = 0.25 * 50.0 / (2.0 * stiffness_5020)
    values["waist_pitch_joint"] = 0.25 * 50.0 / (2.0 * stiffness_5020)
    return np.asarray([values[name] for name in SONIC_POLICY_JOINT_ORDER], dtype=np.float32)


SONIC_ACTION_SCALE = sonic_action_scale()


@dataclass
class SonicG1TrackingCfg(MotionTrackingCfg):
    """Configuration that owns the SONIC 29-DoF observation contract."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_sonic.xml")
        )
    )
    motion_manifest: str | None = None
    motion_rank: int = 0
    motion_world_size: int = 1
    motion_shard_clips: bool = True
    motion_cache_size: int = 2
    motion_global_mmap_sidecar: str | None = None
    motion_global_mmap_trusted_receipt: str | None = None
    # Keep direct construction low-RAM by default. The release MuJoCo owner
    # opts into its rollout-hot field set through the bridge configuration.
    motion_hot_fields: tuple[str, ...] = ()
    motion_verify_checksums: bool = True
    motion_verify_shapes: bool = True
    # SONIC's local-dir history uses active uniform sensor corruption with
    # release scales; the generic G1 profile defaults to level=0.
    noise_config: NoiseConfig = field(
        default_factory=lambda: NoiseConfig(
            level=1.0,
            scale_gravity=0.05,
            scale_gyro=0.2,
            scale_joint_angle=0.01,
            scale_joint_vel=0.5,
        )
    )
    # Upstream ``sonic_release`` uses the pelvis as the motion/robot anchor.
    # The generic UniLab tracking profile historically anchors at ``torso_link``;
    # leaving that inherited default here silently changes every local-frame
    # observation, reward, and termination while preserving all tensor shapes.
    anchor_body_name: str = "pelvis"
    anchor_pos_z_threshold: float = 0.15
    anchor_ori_threshold: float = 0.2
    ee_body_pos_z_threshold: float = 0.15
    root_height_threshold: float = SONIC_TERMINATION_ROOT_HEIGHT_THRESHOLD
    down_height_termination_threshold: float = SONIC_TERMINATION_DOWN_THRESHOLD
    foot_pos_threshold: float = SONIC_TERMINATION_FOOT_POS_THRESHOLD
    root_height_ema_alpha: float = SONIC_TERMINATION_ROOT_HEIGHT_EMA_ALPHA
    truncate_on_clip_end: bool = True
    body_names: tuple[str, ...] = SONIC_BODY_ORDER
    ee_body_names: tuple[str, ...] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    undesired_contact_body_names: tuple[str, ...] = (
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "right_hip_roll_link",
        "right_knee_link",
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
    )
    history_length: int = 10
    num_future_frames: int = 10
    dt_future_ref_frames: float = 0.1
    smpl_num_future_frames: int = 10
    smpl_dt_future_ref_frames: float = 0.02
    smpl_y_up: bool = True
    encoder_names: tuple[str, ...] = ("g1", "teleop", "smpl")
    encoder_sample_probs: tuple[float, ...] = (1.0, 1.0, 1.0)
    teleop_sample_prob_when_smpl: float = 0.5
    tokenizer_enable_corruption: bool = True
    observation_profile: str = SONIC_RELEASE_OBSERVATION_PROFILE
    # Pinned ``motion_lib_cfg.adaptive_sampling`` values from sonic_release.
    # These stay owner-local because the public MotionSampler uses a different
    # global-corpus binning contract.
    adaptive_sampler_bin_size: int = 50
    adaptive_sampler_init_num_failures: float = 1.0
    adaptive_sampler_uniform_sampling_rate: float = 0.1
    adaptive_sampler_failure_rate_max_over_mean: float = 200.0
    adaptive_sampler_pre_failure_sample_window: int = 200
    adaptive_sampler_sequence_length_agnostic: bool = True
    # Update the CDF at PPO-collection cadence, while retaining per-simulation
    # step exposure/failure accounting.  The task owner pins this to its
    # 24-step release rollout horizon rather than leaking it into the runner.
    adaptive_sampler_probability_refresh_interval_steps: int = 24
    # Pinned TrackingCommand default: each training rank only materializes an
    # active view of this many motions, while its bin statistics stay global.
    active_motion_pool_size: int = 1024
    # Pinned ImResampleCallback cadence in completed PPO global steps.
    motion_resample_frequency: int = 250
    # The release task enables this training augmentation in its owner YAML.
    # Leave direct programmatic construction opt-in.
    freeze_frame_aug: bool = False
    freeze_frame_prob: float = 0.1
    vr_body_names: tuple[str, ...] = SONIC_VR_BODY_ORDER
    vr_body_offsets: tuple[tuple[float, float, float], ...] = SONIC_VR_BODY_OFFSETS
    reward_point_body_names: tuple[str, ...] = (
        "torso_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    reward_point_body_offsets: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.5),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    anti_shake_body_names: tuple[str, ...] = (
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "head_link",
    )
    ankle_joint_names: tuple[str, ...] = (
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    )
    use_release_action_scale: bool = True
    # The release wrapper clips the policy sample before passing it to the
    # IsaacLab action manager.  PPO still stores/log-probs the unclipped sample;
    # this bound therefore belongs at the environment action boundary.
    action_clip_value: float = 20.0
    # SONIC clips are materialized in the configured 14-body order. The
    # shared tracking engine otherwise assumes MuJoCo body-id indexing.
    motion_data_body_indices: tuple[int, ...] = tuple(range(len(SONIC_BODY_ORDER)))

    def __post_init__(self) -> None:
        if not 0.0 <= self.freeze_frame_prob <= 1.0:
            raise ValueError(f"freeze_frame_prob must be in [0, 1], got {self.freeze_frame_prob}")
        if self.active_motion_pool_size < 1:
            raise ValueError(
                f"active_motion_pool_size must be positive, got {self.active_motion_pool_size}"
            )
        if self.motion_resample_frequency < 1:
            raise ValueError(
                f"motion_resample_frequency must be positive, got {self.motion_resample_frequency}"
            )
        self.sim_dt = 0.005
        self.ctrl_dt = 0.02
        self.sampling_mode = "adaptive"
        self.sensor.gyro = "pelvis_gyro"
        if self.use_release_action_scale:
            self.control_config.action_scale = SONIC_ACTION_SCALE.copy()


@registry.envcfg("SonicG1Tracking")
@dataclass
class SonicG1TrackingEnvCfg(SonicG1TrackingCfg):
    """Registered SONIC G1 configuration."""


class SonicG1TrackingEnv(MotionTrackingEnv):
    """G1 tracking env exposing actor, critic and tokenizer groups."""

    _cfg: SonicG1TrackingCfg

    def _init_reward_functions(self):
        super()._init_reward_functions()
        body_names = tuple(self._cfg.body_names)
        anti_shake_body_indices: list[int] = []
        for name in self._cfg.anti_shake_body_names:
            if name == "head_link" and name not in body_names:
                anti_shake_body_indices.append(body_names.index("torso_link"))
            else:
                anti_shake_body_indices.append(body_names.index(name))
        self._anti_shake_body_indices = np.asarray(
            anti_shake_body_indices,
            dtype=np.int32,
        )
        self._vr_point_body_indices = np.asarray(
            [body_names.index(name) for name in self._cfg.reward_point_body_names], dtype=np.int32
        )
        self._vr_point_body_offsets = np.asarray(
            self._cfg.reward_point_body_offsets, dtype=np.float32
        )
        if self._vr_point_body_offsets.shape != (len(self._vr_point_body_indices), 3):
            raise ValueError("SONIC reward_point_body_offsets must match reward point bodies")
        self._joint_acc_indices = np.asarray(
            [SONIC_JOINT_ORDER.index(name) for name in self._cfg.ankle_joint_names],
            dtype=np.int32,
        )
        self._reward_fns.update(
            {
                "anti_shake_ang_vel": anti_shake_ang_vel,
                "tracking_vr_5point_local": tracking_vr_5point_local,
                "feet_acc": joint_acc_l2,
            }
        )

    def __init__(self, cfg: SonicG1TrackingCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        if (
            cfg.history_length != 10
            or cfg.num_future_frames != 10
            or cfg.smpl_num_future_frames != 10
        ):
            raise ValueError(
                "SONIC release observations require 10 history, future, and SMPL-future frames"
            )
        if tuple(cfg.encoder_names) != ("g1", "teleop", "smpl"):
            raise ValueError("SONIC release encoder_names must be ('g1', 'teleop', 'smpl')")
        if cfg.observation_profile != SONIC_RELEASE_OBSERVATION_PROFILE:
            raise ValueError(
                "SONIC release observations require "
                f"observation_profile={SONIC_RELEASE_OBSERVATION_PROFILE!r}"
            )
        self._sonic_reset_ids: np.ndarray | None = None
        self._sonic_store = self._resolve_store(cfg)
        # Snapshot immutable motion metadata on the cold path.  Observation
        # construction must only touch frame arrays; it must not repeatedly
        # traverse manifest/asset metadata in ``step``.
        self._sonic_fps = (
            max(1, int(round(self._sonic_store.manifest.clips[0].fps)))
            if self._sonic_store is not None
            else 50
        )
        self._sonic_num_bodies = (
            self._sonic_store.num_bodies if self._sonic_store is not None else len(cfg.body_names)
        )
        self._sonic_has_smpl = bool(
            self._sonic_store is not None
            and {"smpl_joints", "smpl_root_quat_w"}.issubset(self._sonic_store.arrays)
        )
        self._future_offsets = self._future_frame_offsets(
            cfg.num_future_frames, cfg.dt_future_ref_frames, self._sonic_fps
        )
        self._smpl_future_offsets = self._future_frame_offsets(
            cfg.smpl_num_future_frames, cfg.smpl_dt_future_ref_frames, self._sonic_fps
        )
        if self._sonic_store is not None:
            from unilab.training.sonic_store import SonicMotionLoader

            # Inject before the shared tracking owner constructs its loader so
            # both paths retain the bounded, rank-sharded lazy store contract.
            cfg.motion_loader = SonicMotionLoader(self._sonic_store)
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        alpha = float(cfg.root_height_ema_alpha)
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"SONIC root_height_ema_alpha must be in (0, 1], got {alpha}")
        self._running_ref_root_height = np.zeros((num_envs,), dtype=np.float32)
        # ``MotionTrackingEnv`` constructs its general-purpose sampler.  The
        # release curriculum has a different owner-only semantic: bins reset
        # at every clip boundary and clip length must not determine the motion
        # prior.  Replace it after the common owner has finished all generic
        # initialization, before the first reset can sample a reference.
        self.motion_sampler = _SonicReleaseMotionSampler(
            self.motion_loader,
            num_envs=num_envs,
            bin_size=cfg.adaptive_sampler_bin_size,
            init_num_failures=cfg.adaptive_sampler_init_num_failures,
            uniform_sampling_rate=cfg.adaptive_sampler_uniform_sampling_rate,
            failure_rate_max_over_mean=cfg.adaptive_sampler_failure_rate_max_over_mean,
            pre_failure_sample_window=cfg.adaptive_sampler_pre_failure_sample_window,
            sequence_length_agnostic=cfg.adaptive_sampler_sequence_length_agnostic,
            probability_refresh_interval_steps=(
                cfg.adaptive_sampler_probability_refresh_interval_steps
            ),
            active_motion_pool_size=cfg.active_motion_pool_size,
            motion_resample_frequency=cfg.motion_resample_frequency,
            freeze_frame_aug=cfg.freeze_frame_aug,
            freeze_frame_prob=cfg.freeze_frame_prob,
        )
        if self._num_action != len(SONIC_JOINT_ORDER):
            raise ValueError(f"SONIC requires 29 actuators, backend exposes {self._num_action}")
        self._backend_to_policy = self._resolve_actuator_permutation()
        self._policy_to_backend = np.argsort(self._backend_to_policy)
        self._policy_default_angles = self.default_angles[self._backend_to_policy]
        self._policy_joint_range = (
            self._joint_range[self._backend_to_policy] if self._joint_range is not None else None
        )
        if len(cfg.body_names) != 14:
            raise ValueError(
                f"SONIC release observations require 14 bodies, got {len(cfg.body_names)}"
            )
        self._sonic_foot_body_indices = np.asarray(
            [
                cfg.body_names.index("left_ankle_roll_link"),
                cfg.body_names.index("right_ankle_roll_link"),
            ],
            dtype=np.int32,
        )
        if len(cfg.vr_body_names) != 3:
            raise ValueError("SONIC release observations require three VR bodies")
        try:
            self._vr_body_rows = np.asarray(
                [tuple(cfg.body_names).index(name) for name in cfg.vr_body_names], dtype=np.int32
            )
        except ValueError as exc:
            raise ValueError("SONIC VR body names must be present in body_names") from exc
        self._vr_body_offsets = np.asarray(cfg.vr_body_offsets, dtype=np.float32)
        if self._vr_body_offsets.shape != (3, 3):
            raise ValueError("SONIC vr_body_offsets must have shape (3, 3)")
        self._history = np.zeros((num_envs, self._cfg.history_length, 93), dtype=np.float32)
        self._critic_history = np.zeros_like(self._history)
        self._encoder_index = np.zeros((num_envs, len(self._cfg.encoder_names)), dtype=np.float32)
        self._sample_encoder_indices(np.arange(num_envs, dtype=np.int32))
        self._actor_obs_width = SONIC_ACTOR_OBS_DIM
        self._critic_obs_width = SONIC_CRITIC_OBS_DIM

    @staticmethod
    def _resolve_store(cfg: SonicG1TrackingCfg) -> SonicMotionStore | None:
        if not cfg.motion_manifest:
            return None
        from unilab.training.sonic_store import load_sonic_motion_store

        return load_sonic_motion_store(
            cfg.motion_manifest,
            verify_checksums=cfg.motion_verify_checksums,
            verify_shapes=cfg.motion_verify_shapes,
            expected_joint_order=SONIC_JOINT_ORDER,
            expected_body_order=cfg.body_names,
            rank=cfg.motion_rank,
            world_size=cfg.motion_world_size,
            shard_clips=cfg.motion_shard_clips,
            cache_size=cfg.motion_cache_size,
            hot_fields=cfg.motion_hot_fields,
            motion_global_mmap_sidecar=cfg.motion_global_mmap_sidecar,
            motion_global_mmap_trusted_receipt=cfg.motion_global_mmap_trusted_receipt,
        )

    @staticmethod
    def _future_frame_offsets(count: int, spacing: float, fps: int) -> np.ndarray:
        steps = float(spacing) * int(fps)
        rounded_steps = int(round(steps))
        if rounded_steps < 1 or not np.isclose(steps, rounded_steps, atol=1.0e-9):
            raise ValueError(
                f"SONIC future spacing={spacing} at fps={fps} must be a positive integer step"
            )
        return np.arange(count, dtype=np.int64) * rounded_steps

    def _resolve_actuator_permutation(self) -> np.ndarray:
        names = tuple(self._backend.get_actuator_names())
        normalized = tuple(name.removesuffix("_dof") for name in names)
        if set(normalized) != set(SONIC_JOINT_ORDER) or len(normalized) != len(SONIC_JOINT_ORDER):
            raise ValueError(
                "SONIC actuator names do not match the 29-DoF release order: "
                f"expected={SONIC_JOINT_ORDER}, actual={normalized}"
            )
        return np.asarray(
            [normalized.index(name) for name in SONIC_POLICY_JOINT_ORDER], dtype=np.int32
        )

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {
            "actor_obs": SONIC_ACTOR_OBS_DIM,
            "critic_obs": SONIC_CRITIC_OBS_DIM,
            "tokenizer": SONIC_TOKENIZER_OBS_DIM,
        }

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {
                name: gym.spaces.Box(-np.inf, np.inf, shape=(width,), dtype=np.float32)
                for name, width in self.obs_groups_spec.items()
            }
        )

    def _actor_obs_dim(self, n: int) -> int:
        del n
        return SONIC_ACTOR_OBS_DIM

    def _critic_base_obs_dim(self, n: int) -> int:
        del n
        return SONIC_CRITIC_OBS_DIM - len(self._cfg.body_names) * 9

    def _sample_encoder_indices(self, env_ids: np.ndarray) -> None:
        if not len(env_ids):
            return
        probabilities = np.asarray(self._cfg.encoder_sample_probs, dtype=np.float64)
        if probabilities.shape != (len(self._cfg.encoder_names),) or np.any(probabilities < 0):
            raise ValueError("encoder_sample_probs must match encoder_names and be non-negative")
        if probabilities.sum() <= 0:
            raise ValueError("encoder_sample_probs must contain a positive mass")
        teleop_probability = float(self._cfg.teleop_sample_prob_when_smpl)
        if not 0.0 <= teleop_probability <= 1.0:
            raise ValueError("teleop_sample_prob_when_smpl must be in [0, 1]")
        if not self._sonic_has_smpl:
            probabilities[2] = 0.0
            if probabilities.sum() <= 0:
                raise ValueError("encoder_sample_probs need g1 or teleop mass without SMPL data")
        probabilities /= probabilities.sum()
        choices = np.random.choice(len(probabilities), size=len(env_ids), p=probabilities)
        self._encoder_index[env_ids] = 0.0
        self._encoder_index[env_ids, choices] = 1.0
        # Release training uses a multi-hot mask for SMPL-native samples: the
        # paired G1 encoder is always active and teleop is additionally active
        # with the configured probability for latent-alignment losses.
        smpl_ids = env_ids[choices == 2]
        if len(smpl_ids):
            self._encoder_index[smpl_ids, 0] = 1.0
            use_teleop = np.random.random(len(smpl_ids)) < teleop_probability
            self._encoder_index[smpl_ids[use_teleop], 1] = 1.0

    def reset(self, env_indices: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], dict]:
        if env_indices is None:
            if self._state is None:
                state = self.init_state()
                return state.obs, state.info
            env_indices = np.arange(self._num_envs, dtype=np.int32)
        env_indices = np.asarray(env_indices, dtype=np.int32)
        self._sonic_reset_ids = env_indices
        self._sample_encoder_indices(env_indices)
        try:
            obs, info = super().reset(env_indices)
            self._initialize_running_ref_root_height(env_indices)
            return obs, info
        finally:
            self._sonic_reset_ids = None

    def maybe_resample_motion_pool(self, completed_global_steps: int) -> bool:
        """Advance the release-private active-pool callback state.

        A SONIC PPO runner calls this after an optimizer update and performs a
        full :meth:`reset` when it returns true.  This method deliberately does
        not alter the generic environment or runner contracts.
        """

        return self.motion_sampler.maybe_resample_active_motion_pool(completed_global_steps)

    def _initialize_running_ref_root_height(self, env_ids: np.ndarray) -> None:
        if not len(env_ids):
            return
        motion_data = self.motion_loader.get_motion_at_frame(
            self.motion_sampler.current_frames[env_ids]
        )
        self._running_ref_root_height[env_ids] = motion_data.body_pos_w[:, self.anchor_body_idx, 2]

    def _update_running_ref_root_height(self, env_ids: np.ndarray) -> None:
        if not len(env_ids):
            return
        motion_data = self.motion_loader.get_motion_at_frame(
            self.motion_sampler.current_frames[env_ids]
        )
        reference_anchor_z = motion_data.body_pos_w[:, self.anchor_body_idx, 2]
        alpha = float(self._cfg.root_height_ema_alpha)
        self._running_ref_root_height[env_ids] = (
            alpha * reference_anchor_z + (1.0 - alpha) * self._running_ref_root_height[env_ids]
        )

    def update_state(self, state: Any) -> Any:
        previous_frames = self.motion_sampler.timeline_frames.copy()
        previous_clip_end_frames = self.motion_sampler.current_clip_end_frames.copy()
        next_state = super().update_state(state)

        clip_end = previous_frames >= previous_clip_end_frames
        done = next_state.terminated | self._clip_end_truncated
        update_ids = np.flatnonzero(~clip_end & ~done).astype(np.int32)
        self._update_running_ref_root_height(update_ids)

        if not self._cfg.truncate_on_clip_end:
            resampled_ids = np.flatnonzero(clip_end & ~next_state.terminated).astype(np.int32)
            self._initialize_running_ref_root_height(resampled_ids)
        return next_state

    def apply_action(self, actions: np.ndarray, state: Any) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != len(SONIC_JOINT_ORDER):
            raise ValueError(f"SONIC actions must have shape (N, 29), got {actions.shape}")
        clip_value = float(self._cfg.action_clip_value)
        if clip_value <= 0.0:
            raise ValueError(f"SONIC action_clip_value must be positive, got {clip_value}")
        processed_actions = np.clip(actions, -clip_value, clip_value)
        state.info["last_actions"] = state.info.get(
            "current_actions", np.zeros_like(processed_actions)
        )
        state.info["current_actions"] = processed_actions
        delayed = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else processed_actions
        )
        # Scale/defaults share the IsaacLab policy ABI.  Only the completed
        # target is mapped back to the backend/MuJoCo actuator order.
        target_policy = (
            delayed * np.asarray(self._cfg.control_config.action_scale)
            + self._policy_default_angles
        )
        target_backend = target_policy[:, self._policy_to_backend]
        bias = state.info.get("default_dof_pos_bias")
        if isinstance(bias, np.ndarray):
            target_backend = target_backend + bias
        return target_backend

    def _compute_terminations(
        self,
        motion_data: Any,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
    ) -> np.ndarray:
        """Compute the SONIC release tracking termination contract.

        SONIC uses the upstream adaptive height checks and full quaternion
        error.  This owner-level override keeps those semantics isolated from
        the historical generic motion-tracking termination behavior.
        """
        terminated = self._terminated
        terminated.fill(False)

        reference_root_height = self._running_ref_root_height
        anchor_z_error = self._env_error
        np.subtract(
            motion_data.body_pos_w[:, self.anchor_body_idx, 2],
            robot_body_pos_w[:, self.anchor_body_idx, 2],
            out=anchor_z_error,
        )
        np.abs(anchor_z_error, out=anchor_z_error)
        anchor_threshold = np.full_like(anchor_z_error, self._cfg.anchor_pos_z_threshold)
        anchor_threshold[reference_root_height < self._cfg.root_height_threshold] = (
            self._cfg.down_height_termination_threshold
        )
        np.greater(anchor_z_error, anchor_threshold, out=self._env_bool)
        terminated |= self._env_bool

        anchor_ori_error = np_quat_error_magnitude_squared_batched(
            motion_data.body_quat_w[:, self.anchor_body_idx],
            robot_body_quat_w[:, self.anchor_body_idx],
        )
        np.greater(anchor_ori_error, self._cfg.anchor_ori_threshold, out=self._env_bool)
        terminated |= self._env_bool

        if self._has_ee_body_indices:
            np.subtract(
                self.body_pos_relative_w[:, self.ee_body_indices, 2],
                robot_body_pos_w[:, self.ee_body_indices, 2],
                out=self._ee_pos_error_z,
            )
            np.abs(self._ee_pos_error_z, out=self._ee_pos_error_z)
            ee_threshold = np.full_like(self._ee_pos_error_z, self._cfg.ee_body_pos_z_threshold)
            ee_threshold[reference_root_height < self._cfg.root_height_threshold] = (
                self._cfg.down_height_termination_threshold
            )
            np.greater(self._ee_pos_error_z, ee_threshold, out=self._ee_terminated)
            np.logical_or.reduce(self._ee_terminated, axis=1, out=self._env_bool)
            terminated |= self._env_bool

        foot_error = self._body_vec_error[:, self._sonic_foot_body_indices]
        np.subtract(
            self.body_pos_relative_w[:, self._sonic_foot_body_indices],
            robot_body_pos_w[:, self._sonic_foot_body_indices],
            out=foot_error,
        )
        np.square(foot_error, out=foot_error)
        foot_error_squared = self._ee_pos_error_z[:, : self._sonic_foot_body_indices.size]
        np.sum(foot_error, axis=-1, out=foot_error_squared)
        np.greater(
            foot_error_squared,
            self._cfg.foot_pos_threshold**2,
            out=self._ee_terminated[:, : self._sonic_foot_body_indices.size],
        )
        np.logical_or.reduce(
            self._ee_terminated[:, : self._sonic_foot_body_indices.size],
            axis=1,
            out=self._env_bool,
        )
        terminated |= self._env_bool

        if self._cfg.terminate_on_undesired_contacts and self._has_undesired_contact_body_indices:
            body_z = robot_body_pos_w[:, self.undesired_contact_body_indices, 2]
            np.less(
                body_z,
                self._cfg.undesired_contact_z_threshold,
                out=self._undesired_contact_mask,
            )
            np.logical_or.reduce(self._undesired_contact_mask, axis=-1, out=self._env_bool)
            terminated |= self._env_bool

        # Release clip boundaries are command-resampling points.  ``step``
        # performs that resample after this method; treating the final valid
        # frame as a terminal failure would both reset the robot and corrupt
        # the adaptive curriculum statistics.
        return terminated

    def _policy_joint_values(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)[:, self._backend_to_policy]

    def _policy_defaults_for_obs(self, info: Mapping[str, Any], env_ids: np.ndarray) -> np.ndarray:
        defaults = np.broadcast_to(
            np.asarray(self._policy_default_angles, dtype=np.float32),
            (len(env_ids), len(SONIC_JOINT_ORDER)),
        ).copy()
        bias = info.get("default_dof_pos_bias")
        if not isinstance(bias, np.ndarray):
            state = getattr(self, "_state", None)
            state_info = getattr(state, "info", None)
            if isinstance(state_info, Mapping):
                bias = state_info.get("default_dof_pos_bias")
        if bias is None:
            return defaults
        bias = np.asarray(bias, dtype=np.float32)
        if bias.ndim != 2 or bias.shape[1] != len(SONIC_JOINT_ORDER):
            raise ValueError(f"default_dof_pos_bias must have shape (N, 29), got {bias.shape}")
        if bias.shape[0] == self._num_envs:
            bias = bias[env_ids]
        elif bias.shape[0] != len(env_ids):
            raise ValueError(
                "default_dof_pos_bias batch does not match observation rows: "
                f"{bias.shape[0]} versus {len(env_ids)}"
            )
        defaults += bias[:, self._backend_to_policy]
        return defaults

    def _actions_for_obs(self, info: Mapping[str, Any], env_ids: np.ndarray) -> np.ndarray:
        actions = info.get("current_actions")
        if not isinstance(actions, np.ndarray):
            state = getattr(self, "_state", None)
            state_info = getattr(state, "info", None)
            if isinstance(state_info, Mapping):
                actions = state_info.get("current_actions")
        if not isinstance(actions, np.ndarray):
            return np.zeros((len(env_ids), len(SONIC_JOINT_ORDER)), dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != len(SONIC_JOINT_ORDER):
            raise ValueError(f"current_actions must have shape (N, 29), got {actions.shape}")
        if actions.shape[0] == self._num_envs:
            actions = actions[env_ids]
        elif actions.shape[0] != len(env_ids):
            raise ValueError(
                "current_actions batch does not match observation rows: "
                f"{actions.shape[0]} versus {len(env_ids)}"
            )
        return actions

    def _tokenizer_corruption(self, data: np.ndarray, scale: float) -> np.ndarray:
        if not self._cfg.tokenizer_enable_corruption:
            return data
        seed = self._configured_obs_noise_seed()
        rng = np.random if seed is None else getattr(self, "_obs_noise_rng", None)
        if rng is None:
            rng = np.random.default_rng(seed)
            self._obs_noise_rng = rng
        return data + rng.uniform(-scale, scale, data.shape).astype(data.dtype)

    def _future_reference(
        self, frame_indices: np.ndarray, *, env_ids: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        if env_ids is not None:
            env_ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
            if len(env_ids) != len(frame_indices):
                raise ValueError("env_ids must have one entry per reference frame")
        if self._sonic_store is None:
            return self._zero_future_reference(len(frame_indices))
        indices = self._sonic_store.future_indices(frame_indices, self._future_offsets)
        indices = self.motion_sampler.clamp_reference_indices(indices, env_ids)
        flat = indices.reshape(-1)
        future_fields = self._sonic_store.gather_fields(
            ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w"), flat
        )
        result = {
            "joint_pos": np.take(
                future_fields["joint_pos"].reshape(len(frame_indices), -1, 29),
                SONIC_MUJOCO_TO_POLICY,
                axis=-1,
            ),
            "joint_vel": np.take(
                future_fields["joint_vel"].reshape(len(frame_indices), -1, 29),
                SONIC_MUJOCO_TO_POLICY,
                axis=-1,
            ),
            "body_pos": future_fields["body_pos_w"].reshape(
                len(frame_indices), -1, self._sonic_num_bodies, 3
            )[:, :, : len(self._cfg.body_names)],
            "body_quat": future_fields["body_quat_w"].reshape(
                len(frame_indices), -1, self._sonic_num_bodies, 4
            )[:, :, : len(self._cfg.body_names)],
        }
        smpl_indices = self._sonic_store.future_indices(frame_indices, self._smpl_future_offsets)
        smpl_indices = self.motion_sampler.clamp_reference_indices(smpl_indices, env_ids)
        smpl_joint_fields = self._sonic_store.gather_fields(
            ("joint_pos",), smpl_indices.reshape(-1)
        )
        result["smpl_joint_pos"] = np.take(
            smpl_joint_fields["joint_pos"].reshape(len(frame_indices), -1, 29),
            SONIC_MUJOCO_TO_POLICY,
            axis=-1,
        )
        if not self._sonic_has_smpl:
            result.update(self._zero_smpl_reference(len(frame_indices)))
            return result
        smpl_fields = self._sonic_store.gather_fields(
            ("smpl_joints", "smpl_root_quat_w"), smpl_indices.reshape(-1)
        )
        result.update(
            {
                "smpl_joints": smpl_fields["smpl_joints"].reshape(len(frame_indices), -1, 24, 3),
                "smpl_root_quat": smpl_fields["smpl_root_quat_w"].reshape(
                    len(frame_indices), -1, 4
                ),
            }
        )
        return result

    def _observation_frame_indices(
        self, env_ids: np.ndarray, *, reference_refresh: bool
    ) -> np.ndarray:
        frame_indices = self.motion_sampler.current_frames[env_ids].copy()
        if self._sonic_reset_ids is None and not reference_refresh:
            np.add(frame_indices, 1, out=frame_indices)
            np.minimum(
                frame_indices,
                self.motion_sampler.current_clip_end_frames[env_ids],
                out=frame_indices,
            )
        # A frozen reference remains at its selected frame even when the
        # observation path asks for the next frame.  The sampler owns this
        # clip-local mapping and keeps it out of the shared tracking engine.
        return self.motion_sampler.clamp_reference_indices(frame_indices, env_ids)

    def _zero_smpl_reference(self, num_envs: int) -> dict[str, np.ndarray]:
        smpl_frames = self._cfg.smpl_num_future_frames
        return {
            "smpl_joints": np.zeros((num_envs, smpl_frames, 24, 3), dtype=np.float32),
            "smpl_root_quat": np.broadcast_to(
                np.asarray([1, 0, 0, 0], dtype=np.float32),
                (num_envs, smpl_frames, 4),
            ).copy(),
        }

    def _zero_future_reference(self, num_envs: int) -> dict[str, np.ndarray]:
        frames = self._cfg.num_future_frames
        result = {
            "joint_pos": np.zeros((num_envs, frames, 29), dtype=np.float32),
            "joint_vel": np.zeros((num_envs, frames, 29), dtype=np.float32),
            "body_pos": np.zeros(
                (num_envs, frames, len(self._cfg.body_names), 3), dtype=np.float32
            ),
            "body_quat": np.broadcast_to(
                np.asarray([1, 0, 0, 0], dtype=np.float32),
                (num_envs, frames, len(self._cfg.body_names), 4),
            ).copy(),
            "smpl_joint_pos": np.zeros(
                (num_envs, self._cfg.smpl_num_future_frames, 29), dtype=np.float32
            ),
        }
        result.update(self._zero_smpl_reference(num_envs))
        return result

    def _build_history(
        self,
        env_ids: np.ndarray,
        linvel: np.ndarray,
        gyro: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        robot_body_quat_w: np.ndarray,
        last_actions: np.ndarray,
        *,
        policy_default_angles: np.ndarray,
        advance_history: bool,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        dof_pos = self._policy_joint_values(dof_pos)
        dof_vel = self._policy_joint_values(dof_vel)
        anchor_quat = robot_body_quat_w[:, self.anchor_body_idx]
        gravity = np.broadcast_to(np.asarray([0.0, 0.0, -1.0], dtype=np.float32), (len(env_ids), 3))
        gravity = np.asarray(np_quat_apply_inverse(anchor_quat, gravity), dtype=np.float32)
        joint_pos = dof_pos - policy_default_angles
        noise_cfg = self._cfg.noise_config
        actor_gyro = gyro
        actor_joint_pos = joint_pos
        actor_joint_vel = dof_vel
        actor_gravity = gravity
        if noise_cfg.level > 0:
            actor_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
            actor_joint_pos = self._obs_noise(joint_pos, noise_cfg.scale_joint_angle)
            actor_joint_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
            actor_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        actor_current = np.concatenate(
            [actor_gyro, actor_joint_pos, actor_joint_vel, last_actions, actor_gravity], axis=1
        )
        critic_current = np.concatenate([linvel, gyro, joint_pos, dof_vel, last_actions], axis=1)
        reset_ids = self._sonic_reset_ids
        if reset_ids is not None or not advance_history:
            self._history[env_ids] = actor_current[:, None, :]
            self._critic_history[env_ids] = critic_current[:, None, :]
        else:
            self._history[env_ids, :-1] = self._history[env_ids, 1:]
            self._history[env_ids, -1] = actor_current
            self._critic_history[env_ids, :-1] = self._critic_history[env_ids, 1:]
            self._critic_history[env_ids, -1] = critic_current
        actor_history = self._history[env_ids]
        critic_history = self._critic_history[env_ids]
        return (
            {
                "base_ang_vel": actor_history[:, :, 0:3],
                "joint_pos": actor_history[:, :, 3:32],
                "joint_vel": actor_history[:, :, 32:61],
                "actions": actor_history[:, :, 61:90],
                "gravity_dir": actor_history[:, :, 90:93],
            },
            {
                "base_lin_vel": critic_history[:, :, 0:3],
                "base_ang_vel": critic_history[:, :, 3:6],
                "joint_pos": critic_history[:, :, 6:35],
                "joint_vel": critic_history[:, :, 35:64],
                "actions": critic_history[:, :, 64:93],
            },
        )

    def _compute_obs(
        self,
        info: dict,
        motion_data: Any,
        linvel: np.ndarray,
        gyro: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
    ) -> dict[str, np.ndarray]:
        del motion_data
        env_ids = np.asarray(info.get("env_ids", np.arange(linvel.shape[0])), dtype=np.int32)
        if env_ids.shape[0] != linvel.shape[0]:
            env_ids = np.arange(linvel.shape[0], dtype=np.int32)
        policy_default_angles = self._policy_defaults_for_obs(info, env_ids)
        last_actions = self._actions_for_obs(info, env_ids)
        is_reference_refresh = "env_ids" in info
        is_clip_refresh = is_reference_refresh and self._sonic_reset_ids is None
        if is_clip_refresh:
            self._sample_encoder_indices(env_ids)
        actor_terms, critic_history_terms = self._build_history(
            env_ids,
            linvel,
            gyro,
            dof_pos,
            dof_vel,
            robot_body_quat_w,
            last_actions,
            policy_default_angles=policy_default_angles,
            advance_history=not is_clip_refresh,
        )

        frame_indices = self._observation_frame_indices(
            env_ids, reference_refresh=is_reference_refresh
        )
        future = self._future_reference(frame_indices, env_ids=env_ids)
        robot_anchor_pos = robot_body_pos_w[:, self.anchor_body_idx]
        robot_anchor_quat = robot_body_quat_w[:, self.anchor_body_idx]
        ref_body_pos = future["body_pos"][:, 0]
        ref_body_quat = future["body_quat"][:, 0]
        anchor_pos = ref_body_pos[:, self.anchor_body_idx]
        anchor_quat = ref_body_quat[:, self.anchor_body_idx]
        anchor_pos_b = np.empty((len(env_ids), 3), dtype=np.float32)
        anchor_ori_b = np.empty((len(env_ids), 6), dtype=np.float32)
        np_write_relative_anchor_transform_pos_rot6d(
            robot_anchor_pos,
            robot_anchor_quat,
            anchor_pos,
            anchor_quat,
            anchor_pos_b,
            anchor_ori_b,
        )

        body_pos_b = np_quat_apply_batched(
            np_quat_conjugate_batched(robot_anchor_quat)[:, None, :],
            robot_body_pos_w - robot_anchor_pos[:, None, :],
        ).astype(np.float32)
        body_ori_b = np_matrix_first_two_cols_from_quat(
            np_quat_mul_batched(
                np.broadcast_to(
                    np_quat_conjugate_batched(robot_anchor_quat)[:, None, :],
                    robot_body_quat_w.shape,
                ),
                robot_body_quat_w,
            )
        ).astype(np.float32)

        ref_anchor_quat = future["body_quat"][:, :, self.anchor_body_idx]
        relative_future_quat = np_quat_mul_batched(
            np.broadcast_to(
                np_quat_conjugate_batched(robot_anchor_quat)[:, None, :],
                ref_anchor_quat.shape,
            ),
            ref_anchor_quat,
        )
        future_ori_b = np_matrix_first_two_cols_from_quat(relative_future_quat).reshape(
            len(env_ids), -1
        )
        command = np.concatenate(
            [
                future["joint_pos"].reshape(len(env_ids), -1),
                future["joint_vel"].reshape(len(env_ids), -1),
            ],
            axis=1,
        )
        command_z_multi = future["body_pos"][:, :, self.anchor_body_idx, 2:3]
        lower = np.concatenate(
            [
                future["joint_pos"][:, :, SONIC_LOWER_BODY_POLICY_INDICES].reshape(
                    len(env_ids), -1
                ),
                future["joint_vel"][:, :, SONIC_LOWER_BODY_POLICY_INDICES].reshape(
                    len(env_ids), -1
                ),
            ],
            axis=1,
        )
        vr_rows = self._vr_body_rows
        vr_pos_w = future["body_pos"][:, 0, vr_rows] + np_quat_apply_batched(
            future["body_quat"][:, 0, vr_rows], self._vr_body_offsets[None, :, :]
        )
        vr_pos = vr_pos_w - anchor_pos[:, None, :]
        vr_pos = np_quat_apply_batched(
            np_quat_conjugate_batched(anchor_quat)[:, None, :], vr_pos
        ).reshape(len(env_ids), -1)
        vr_quat = future["body_quat"][:, 0, vr_rows]
        vr_quat = np_quat_mul_batched(
            np.broadcast_to(np_quat_conjugate_batched(anchor_quat)[:, None, :], vr_quat.shape),
            vr_quat,
        ).reshape(len(env_ids), -1)
        smpl_joints = future["smpl_joints"]
        smpl_root = future["smpl_root_quat"]
        smpl_local = np_quat_apply_batched(
            np_quat_conjugate_batched(smpl_root)[..., None, :], smpl_joints
        )
        smpl_ori_b = np_matrix_first_two_cols_from_quat(
            np_quat_mul_batched(
                np.broadcast_to(
                    np_quat_conjugate_batched(robot_anchor_quat)[:, None, :],
                    smpl_root.shape,
                ),
                smpl_root,
            )
        )
        # TokenizerCfg.enable_corruption is independent of actor noise level.
        anchor_ori_token = self._tokenizer_corruption(anchor_ori_b, 0.05)
        future_ori_token = self._tokenizer_corruption(future_ori_b, 0.05)
        smpl_local = self._tokenizer_corruption(smpl_local, 0.05)
        smpl_ori_token = self._tokenizer_corruption(smpl_ori_b, 0.05)
        wrist_q = future["smpl_joint_pos"][:, :, SONIC_WRIST_JOINT_INDICES]
        actor_obs = pack_sonic_observation_terms(actor_terms, SONIC_ACTOR_OBSERVATION_TERMS)
        critic_obs = pack_sonic_observation_terms(
            {
                "command_multi_future": command,
                "motion_anchor_pos_b": anchor_pos_b,
                "motion_anchor_ori_b": anchor_ori_b,
                "body_pos": body_pos_b,
                "body_ori": body_ori_b,
                **critic_history_terms,
            },
            SONIC_CRITIC_OBSERVATION_TERMS,
        )
        tokenizer = pack_sonic_observation_terms(
            {
                "encoder_index": self._encoder_index[env_ids],
                "command_multi_future_nonflat": command.reshape(len(env_ids), 10, 58),
                "command_z_multi_future_nonflat": command_z_multi,
                "command_z": command_z_multi[:, 0],
                "motion_anchor_ori_b": anchor_ori_token,
                "motion_anchor_ori_b_mf_nonflat": future_ori_token.reshape(len(env_ids), 10, 6),
                "command_multi_future_lower_body": lower,
                "vr_3point_local_target": vr_pos,
                "vr_3point_local_orn_target": vr_quat,
                "smpl_joints_multi_future_local_nonflat": smpl_local.reshape(len(env_ids), 10, 72),
                "smpl_root_ori_b_multi_future": smpl_ori_token,
                "joint_pos_multi_future_wrist_for_smpl": wrist_q,
            },
            SONIC_TOKENIZER_OBSERVATION_TERMS,
        )
        return {
            "actor_obs": actor_obs,
            "critic_obs": critic_obs,
            "tokenizer": tokenizer,
        }


@registry.env("SonicG1Tracking", sim_backend="mujoco")
class RegisteredSonicG1TrackingEnv(SonicG1TrackingEnv):
    """Registry binding kept separate so the implementation remains testable."""


__all__ = [
    "SONIC_ACTION_SCALE",
    "SONIC_BODY_ORDER",
    "SONIC_JOINT_ORDER",
    "SONIC_LOWER_BODY_POLICY_INDICES",
    "SONIC_MUJOCO_TO_POLICY",
    "SONIC_POLICY_JOINT_ORDER",
    "SONIC_POLICY_TO_MUJOCO",
    "SONIC_RELEASE_OBSERVATION_PROFILE",
    "SONIC_RELEASE_REVISION",
    "SONIC_TOKENIZER_OBSERVATION_TERMS",
    "SONIC_WRIST_JOINT_INDICES",
    "SonicObservationTerm",
    "SonicG1TrackingCfg",
    "SonicG1TrackingEnv",
    "SonicG1TrackingEnvCfg",
    "pack_sonic_observation_terms",
    "sonic_action_scale",
]
