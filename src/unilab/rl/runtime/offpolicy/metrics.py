"""Device-resident collector metric aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class MetricFlush:
    """One host transfer of completed episode and done-count metrics."""

    rewards: list[float]
    lengths: list[int]
    done_count: int
    timeout_count: int
    reward_component_means: dict[str, float] | None = None
    reward_component_updates: int = 0


class CollectorMetricState:
    """Accumulate collector metrics and flush them in one D2H transfer.

    A fixed ``interval * num_envs`` window removes per-step scalar ``.item()``,
    ``.tolist()``, and dynamic ``nonzero()`` host synchronization. Compaction is
    performed on device once per flush; only the compacted result reaches host.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        interval: int,
        device: str | torch.device,
        reward_component_names: Sequence[str] = (),
        storage_device: str | torch.device | None = None,
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("collector metrics require a positive environment count")
        if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
            raise ValueError("collector metrics require a positive interval")
        self.num_envs = int(num_envs)
        self.interval = int(interval)
        self.device = torch.device(device)
        # CPU acceptance changes physical placement only; source logical device
        # remains configuration and is strictly compared by the restore core.
        placement = self.device if storage_device is None else torch.device(storage_device)
        self.reward_component_names = tuple(reward_component_names)
        self.reward_component_count = len(self.reward_component_names)
        self._reward_component_sum = (
            torch.zeros(self.reward_component_count, dtype=torch.float64, device=placement)
            if self.reward_component_count
            else None
        )
        self._reward_component_updates = 0
        self.current_rewards = torch.zeros(self.num_envs, device=placement)
        self.current_lengths = torch.zeros(self.num_envs, dtype=torch.int64, device=placement)
        self.finished_rewards = torch.zeros(
            (self.interval, self.num_envs), dtype=torch.float64, device=placement
        )
        self.finished_lengths = torch.zeros(
            (self.interval, self.num_envs), dtype=torch.float64, device=placement
        )
        self.finished_valid = torch.zeros(
            (self.interval, self.num_envs), dtype=torch.bool, device=placement
        )
        self.done_count = torch.zeros((), dtype=torch.int64, device=placement)
        self.timeout_count = torch.zeros((), dtype=torch.int64, device=placement)
        self._cursor = 0
        self._updates = 0
        self._finalized = False

    def restore_continuation_values(self, saved):
        if set(saved) != set(vars(self)):
            raise ValueError("Collector metric state fields changed")
        for key, value in saved.items():
            target = getattr(self, key)
            if isinstance(value, torch.Tensor):
                if (
                    not isinstance(target, torch.Tensor)
                    or target.shape != value.shape
                    or target.dtype != value.dtype
                ):
                    raise ValueError(f"Collector metric tensor {key} changed")
                target.copy_(value.to(target.device))
            elif key in {
                "device",
                "num_envs",
                "interval",
                "reward_component_names",
                "reward_component_count",
            }:
                if value != target:
                    raise ValueError(f"Collector metric configuration {key} changed")
            else:
                setattr(self, key, value)


class CollectorMetrics(CollectorMetricState):
    """Device aggregation using the same cold value carrier as CPU acceptance."""

    @property
    def ready(self) -> bool:
        return self._updates >= self.interval

    def update(
        self,
        rewards: torch.Tensor,
        done: torch.Tensor,
        timeout: torch.Tensor,
        reward_components: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(rewards, torch.Tensor):
            raise TypeError("collector metric rewards must be Torch tensors")
        if not rewards.is_floating_point():
            raise TypeError("collector metric rewards must have a floating dtype")
        if not isinstance(done, torch.Tensor) or not isinstance(timeout, torch.Tensor):
            raise TypeError("collector metric done and timeout values must be Torch tensors")
        if rewards.shape != (self.num_envs,) or done.shape != rewards.shape:
            raise ValueError("collector metric values must have environment shape")
        if timeout.shape != done.shape:
            raise ValueError("collector metric timeout values must have environment shape")
        if done.dtype != torch.bool or timeout.dtype != torch.bool:
            raise TypeError("collector metric done and timeout values must be boolean tensors")
        if (
            rewards.device != self.device
            or done.device != self.device
            or timeout.device != self.device
        ):
            raise ValueError("collector metric values must live on one device")
        if self.reward_component_count:
            if not isinstance(reward_components, torch.Tensor):
                raise TypeError("collector metric reward components must be a Torch tensor")
            expected_components = (self.num_envs, self.reward_component_count)
            if reward_components.shape != expected_components:
                raise ValueError(
                    f"collector metric reward components must have shape {expected_components}"
                )
            if reward_components.dtype != torch.float32 or not reward_components.is_contiguous():
                raise TypeError("collector metric reward components must be contiguous float32")
            if reward_components.device != self.device:
                raise ValueError(f"collector metric reward components must live on {self.device}")
        elif reward_components is not None:
            raise ValueError("collector metric has no reward component names")
        if self._updates >= self.interval:
            raise RuntimeError("collector metric window was not flushed before reuse")

        episode_rewards = self.current_rewards + rewards
        episode_lengths = self.current_lengths + 1
        row = self._cursor
        self.finished_rewards[row].copy_(episode_rewards)
        self.finished_lengths[row].copy_(episode_lengths)
        self.finished_valid[row].copy_(done)
        self.current_rewards = torch.where(done, torch.zeros_like(episode_rewards), episode_rewards)
        self.current_lengths = torch.where(done, torch.zeros_like(episode_lengths), episode_lengths)
        self.done_count += done.sum()
        self.timeout_count += timeout.sum()
        if self._reward_component_sum is not None and reward_components is not None:
            self._reward_component_sum += reward_components.mean(dim=0, dtype=torch.float64)
            self._reward_component_updates += 1
        self._cursor = (row + 1) % self.interval
        self._updates += 1

    def flush(self) -> MetricFlush:
        if self._finalized:
            raise RuntimeError("collector metric window was already finalized")
        if not self.ready:
            raise RuntimeError("collector metric window is not ready")
        return self._flush()

    def final_flush(self) -> MetricFlush:
        """Flush completed episodes once at collector shutdown.

        A partial shutdown flush uses the same device-side compaction and single
        packed host transfer as the periodic path.  Unfinished episode
        accumulators remain in ``current_rewards`` and ``current_lengths`` and
        are therefore never published as completed episodes.  Calling this
        method again is idempotent and performs no additional device transfer.
        """
        if self._finalized:
            return MetricFlush([], [], 0, 0)
        self._finalized = True
        if self._updates == 0:
            return MetricFlush([], [], 0, 0)
        return self._flush()

    def _flush(self) -> MetricFlush:
        valid = self.finished_valid.reshape(-1)
        rewards = self.finished_rewards.reshape(-1)[valid]
        lengths = self.finished_lengths.reshape(-1)[valid]
        # Float64 exactly represents the bounded integer lengths and counts,
        # while rewards retain at least their accumulated float32 precision.
        # Concatenation keeps episode values and scalar counters to one D2H.
        host_values = torch.cat(
            (
                rewards,
                lengths,
                self._reward_component_sum
                if self._reward_component_sum is not None
                else torch.empty(0, dtype=torch.float64, device=self.device),
                self.done_count.reshape(1),
                self.timeout_count.reshape(1),
            )
        ).cpu()
        episode_count = int(rewards.numel())
        reward_values = host_values[:episode_count].tolist()
        component_tail = self.reward_component_count + 2
        length_values = (
            host_values[episode_count:-component_tail].tolist()
            if self.reward_component_count
            else host_values[episode_count:-2].tolist()
        )
        component_values = (
            host_values[-(self.reward_component_count + 2) : -2].tolist()
            if self.reward_component_count
            else []
        )
        result = MetricFlush(
            rewards=[float(value) for value in reward_values],
            lengths=[int(value) for value in length_values],
            done_count=int(host_values[-2].item()),
            timeout_count=int(host_values[-1].item()),
            reward_component_means=(
                {
                    name: float(value) / max(self._reward_component_updates, 1)
                    for name, value in zip(
                        self.reward_component_names, component_values, strict=True
                    )
                }
                if self._reward_component_sum is not None
                else None
            ),
            reward_component_updates=self._reward_component_updates,
        )
        self.finished_rewards.zero_()
        self.finished_lengths.zero_()
        self.finished_valid.zero_()
        self.done_count.zero_()
        self.timeout_count.zero_()
        if self._reward_component_sum is not None:
            self._reward_component_sum.zero_()
        self._reward_component_updates = 0
        self._cursor = 0
        self._updates = 0
        return result


__all__ = ["CollectorMetricState", "CollectorMetrics", "MetricFlush"]
