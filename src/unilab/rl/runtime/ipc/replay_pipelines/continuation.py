"""Device-independent replay values; execution handles remain with the device owner.

The production device pipeline and CPU acceptance carrier use identical cold
numeric initialization and restoration. CPU carriers cannot execute training.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from unilab.rl.runtime.ipc.replay_buffer import ReplayBuffer
from unilab.rl.runtime.ipc.replay_pipelines.base import ReplayTickMetadata


def initialize_replay_continuation(
    target, replay_buffer, *, sample_count: int, base_seed: int
) -> None:
    """Initialize the same numeric state before device or host storage allocation."""
    target._replay_buffer = replay_buffer
    target._sample_count = int(sample_count)
    target._base_seed = int(base_seed)
    target._capacity = int(replay_buffer.capacity)
    target._storage_width = int(replay_buffer.storage_width)
    target._packed_width = target._storage_width
    target._submitted_ptr = 0
    target._visible_ptr = 0
    target._ingress_high_water = replay_buffer.ingress_diagnostics()["high_water_occupancy"]
    target._hot = 0
    target._cold = 1
    target._has_hot_batch = False
    target._hot_metadata = None
    target._prepared_metadata = None
    target._prepare_tick_id = None
    target._prepare_required_ptr = 0
    target._prepare_state = "idle"


def restore_replay_continuation_values(self, state: dict[str, Any]) -> list:
    """Validate/copy saved values into cold allocated storage, without sampling.

    The device owner must hold its pause before entering and synchronize/create
    new completion events afterward. Returned spans contain values, not events.
    """
    if state.get("schema") != "unilab.device-replay-continuation.v1":
        raise ValueError("Unsupported device replay continuation schema")
    if state.get("layout") != [self._capacity, self._storage_width, self._sample_count]:
        raise ValueError("Replay continuation layout differs")
    if state.get("sampler") != self.continuation_sampler_configuration():
        raise ValueError("Replay continuation sampler differs")
    if int(state["base_seed"]) != self._base_seed:
        raise ValueError("Replay continuation seed differs")
    if self._submitted_ptr or self._visible_ptr or self._prepare_tick_id is not None:
        raise RuntimeError("Replay continuation restore requires an unused pipeline")
    submitted, visible = int(state["submitted_ptr"]), int(state["visible_ptr"])
    ingress = state["ingress"]
    if not 0 <= visible <= submitted <= int(ingress["published_ptr"]):
        raise ValueError("Invalid replay continuation pointer order")
    if visible != int(ingress["ptr"]):
        raise ValueError("Replay continuation visible/committed pointers differ")
    hot, cold = int(state["hot_slot"]), int(state["cold_slot"])
    if {hot, cold} != {0, 1}:
        raise ValueError("Invalid replay continuation slots")
    prepare_state = state["prepare_state"]
    if prepare_state not in {"idle", "preparing", "gather_submitted", "ready"}:
        raise ValueError("Invalid replay continuation prepare state")
    prepared = (
        ReplayTickMetadata(**state["prepared_metadata"])
        if state["prepared_metadata"] is not None
        else None
    )
    hot_metadata = (
        ReplayTickMetadata(**state["hot_metadata"]) if state["hot_metadata"] is not None else None
    )
    if prepared is not None and (
        prepared.batch_gpu_slot != cold
        or prepared.tick_id != state["prepare_tick_id"]
        or prepared.sample_count != self._sample_count
        or prepared.sample_seed != self._base_seed + prepared.tick_id
    ):
        raise ValueError("Invalid prepared replay continuation metadata")
    if bool(state["has_hot_batch"]) != (hot_metadata is not None):
        raise ValueError("Invalid hot replay continuation metadata")
    storage = state["storage"]
    if (
        not isinstance(storage, torch.Tensor)
        or storage.dtype != torch.float32
        or tuple(storage.shape) != (min(submitted, self._capacity), self._storage_width)
    ):
        raise ValueError("Invalid replay continuation ring tensor")
    slots = state["slots"]
    if len(slots) != 2:
        raise ValueError("Invalid replay continuation slot count")
    for index, slot in enumerate(slots):
        live = (prepared is not None and index == cold) or (
            bool(state["has_hot_batch"]) and index == hot
        )
        if live and slot is None:
            raise ValueError("Replay continuation is missing a prefetched batch")
        if slot is not None and (
            not isinstance(slot, torch.Tensor)
            or slot.dtype != torch.float32
            or tuple(slot.shape) != (self._sample_count, self._packed_width)
        ):
            raise ValueError("Invalid replay continuation batch tensor")
    expected_start = visible
    pending_spans = state["pending_spans"]
    if len(pending_spans) != int(ingress["consumed_sequence"]) - int(ingress["released_sequence"]):
        raise ValueError("Replay continuation pending span/ingress count differs")
    for offset, (end, slot, start, count) in enumerate(pending_spans):
        pending = ingress["pending"][offset]
        if (
            int(start) != expected_start
            or int(end) != int(start) + int(count)
            or int(slot) != pending["slot"]
            or int(start) != pending["start"]
            or int(count) != pending["count"]
        ):
            raise ValueError("Invalid replay continuation pending span")
        expected_start = int(end)
    if expected_start != submitted:
        raise ValueError("Replay continuation spans do not reach submitted pointer")

    self._replay_buffer.import_continuation_state(ingress)
    self._gpu_storage[: storage.shape[0]].copy_(storage)
    for index, slot in enumerate(slots):
        if slot is not None:
            self._gpu_packed[index].copy_(slot)
    self._submitted_ptr, self._visible_ptr = submitted, visible
    self._hot, self._cold = hot, cold
    self._has_hot_batch = bool(state["has_hot_batch"])
    self._hot_metadata, self._prepared_metadata = hot_metadata, prepared
    self._prepare_tick_id = state["prepare_tick_id"]
    self._prepare_required_ptr = int(state["prepare_required_ptr"])
    self._prepare_state = prepare_state
    self._ingress_high_water = int(state["ingress_high_water"])
    return [
        [int(end), int(slot), int(start), int(count)] for end, slot, start, count in pending_spans
    ]


class HostReplayContinuation:
    """CPU acceptance carrier allocated from the frozen configuration.

    Logical device and sampler metadata describe the source run. Only tensor
    placement changes to CPU; no checkpoint tensor creates or sizes the target.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        *,
        sample_count: int,
        base_seed: int,
        logical_device: str | torch.device,
        sampler: dict[str, Any],
    ) -> None:
        initialize_replay_continuation(
            self, replay_buffer, sample_count=sample_count, base_seed=base_seed
        )
        self.logical_device = torch.device(logical_device)
        self._sampler = dict(sampler)
        self._gpu_storage = torch.empty((self._capacity, self._storage_width), dtype=torch.float32)
        self._gpu_packed = [
            torch.empty((self._sample_count, self._packed_width), dtype=torch.float32)
            for _ in range(2)
        ]
        self.pending_spans = []

    def continuation_sampler_configuration(self) -> dict[str, Any]:
        return dict(self._sampler)

    def import_continuation_state(self, state: dict[str, Any]) -> None:
        self.pending_spans = restore_replay_continuation_values(self, state)

    def continuation_value_views(self) -> dict[str, Any]:
        """Observe live target values without exporting a second large copy."""
        live_slots = set()
        if self._has_hot_batch:
            live_slots.add(self._hot)
        if self._prepared_metadata is not None:
            live_slots.add(self._prepared_metadata.batch_gpu_slot)
        return {
            "schema": "unilab.device-replay-continuation.v1",
            "layout": [self._capacity, self._storage_width, self._sample_count],
            "sampler": self.continuation_sampler_configuration(),
            "storage": self._gpu_storage[: min(self._submitted_ptr, self._capacity)],
            "slots": [
                slot if index in live_slots else None for index, slot in enumerate(self._gpu_packed)
            ],
            "ingress": self._replay_buffer.export_continuation_state(),
            "submitted_ptr": self._submitted_ptr,
            "visible_ptr": self._visible_ptr,
            "pending_spans": self.pending_spans,
            "hot_slot": self._hot,
            "cold_slot": self._cold,
            "has_hot_batch": self._has_hot_batch,
            "hot_metadata": asdict(self._hot_metadata) if self._hot_metadata is not None else None,
            "prepared_metadata": asdict(self._prepared_metadata)
            if self._prepared_metadata is not None
            else None,
            "prepare_tick_id": self._prepare_tick_id,
            "prepare_required_ptr": self._prepare_required_ptr,
            "prepare_state": self._prepare_state,
            "base_seed": self._base_seed,
            "ingress_high_water": self._ingress_high_water,
        }
