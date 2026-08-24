"""Environment loop and checkpoint handling for SONIC PPO."""

from __future__ import annotations

import hashlib
import inspect
import os
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from unilab.training.sonic_contract import validate_sonic_owner
from unilab.training.sonic_resources import apply_sonic_torch_threads

from .algorithm import SonicPPO
from .model import SonicActorCritic
from .storage import SonicRolloutStorage


def _get(cfg: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = cfg
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _obs(value: Any, key: str, dim: int, device: torch.device, batch: int) -> torch.Tensor:
    if isinstance(value, Mapping):
        aliases = {
            "actor": ("actor_obs", "actor", "policy", "obs"),
            "critic": ("critic_obs", "critic", "privileged", "value", "obs", "actor_obs"),
            "tokenizer": (
                "tokenizer_obs",
                "tokenizer",
                "tokens",
                "reference",
                "obs",
                "actor",
            ),
        }[key]
        value = next((value[name] for name in aliases if name in value), None)
        if value is None:
            raise ValueError(f"observation mapping does not contain a {key} group")
    tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0).expand(batch, -1)
    if tensor.ndim != 2 or tensor.shape[0] != batch:
        raise ValueError(
            f"{key} observation expects batch shape ({batch}, {dim}), got {tuple(tensor.shape)}"
        )
    if tensor.shape[-1] != dim:
        raise ValueError(f"{key} observation expects {dim} features, got {tensor.shape[-1]}")
    return tensor


def _batch_bool(value: Any, *, name: str, device: torch.device, batch: int) -> torch.Tensor:
    """Convert an environment termination field to a checked batch mask."""

    tensor = torch.as_tensor(value, device=device, dtype=torch.bool)
    if tensor.ndim == 0:
        tensor = tensor.expand(batch)
    else:
        tensor = tensor.reshape(-1)
    if tensor.shape != (batch,):
        raise ValueError(
            f"{name} termination mask expects {batch} values, got {tuple(tensor.shape)}"
        )
    return tensor


def _rollout_snapshot(value: torch.Tensor) -> torch.Tensor:
    """Detach CPU observations before an env can mutate its backing buffer.

    ``torch.as_tensor`` intentionally avoids a copy for CPU NumPy observations.
    Some vectorized environments reuse that array in ``step``; the rollout
    must retain the pre-action observation in that case. GPU tensors already
    own device storage and do not need the extra per-step copy.
    """

    return value.clone() if value.device.type == "cpu" else value


def _synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _broadcast_model_state(model: torch.nn.Module) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    with torch.no_grad():
        for tensor in (*model.parameters(), *model.buffers()):
            dist.broadcast(tensor, src=0)


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_distributed_checkpoint(path: Path) -> None:
    """Require every rank to observe the same complete checkpoint file."""

    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        local_result = {"digest": _checkpoint_sha256(path), "error": None}
    except OSError as exc:
        local_result = {"digest": None, "error": f"{type(exc).__name__}: {exc}"}
    gathered: list[dict[str, str | None] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_result)
    unreadable = {
        rank: None if result is None else result["error"]
        for rank, result in enumerate(gathered)
        if result is None or result["error"] is not None
    }
    if unreadable:
        raise ValueError(f"SONIC checkpoint is unreadable on distributed rank(s): {unreadable}")
    digests = {str(result["digest"]) for result in gathered if result is not None}
    if len(digests) != 1:
        raise ValueError("SONIC checkpoint contents differ across distributed ranks")


def _synchronize_checkpoint_load(*, error: BaseException | None, iteration: int | None) -> None:
    """Make checkpoint deserialization success/failure identical on every rank."""

    if not (dist.is_available() and dist.is_initialized()):
        return
    local_result = {
        "error": None if error is None else f"{type(error).__name__}: {str(error)[:1000]}",
        "iteration": iteration,
    }
    gathered: list[dict[str, str | int | None] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_result)
    failures = {
        rank: "rank did not report a checkpoint result" if result is None else result["error"]
        for rank, result in enumerate(gathered)
        if result is None or result["error"] is not None
    }
    if failures:
        raise ValueError(f"SONIC checkpoint load failed on distributed rank(s): {failures}")
    iterations = {result["iteration"] for result in gathered if result is not None}
    if len(iterations) != 1:
        ordered = sorted(iterations, key=lambda value: repr(value))
        raise ValueError(f"SONIC checkpoint iterations differ across ranks: {ordered}")


def _finish_sonic_distributed(*, owned: bool) -> None:
    if not owned or not (dist.is_available() and dist.is_initialized()):
        return
    dist.destroy_process_group()


def _cleanup_sonic_runtime(
    *, runner: SonicPPORunner | None, env: Any, owned_process_group: bool, suppress_errors: bool
) -> None:
    failures: list[BaseException] = []
    if runner is not None:
        try:
            runner._finish_logger(suppress_errors=False)
        except BaseException as exc:
            failures.append(exc)
    close = getattr(env, "close", None)
    if callable(close):
        try:
            close()
        except BaseException as exc:
            failures.append(exc)
    try:
        _finish_sonic_distributed(owned=owned_process_group)
    except BaseException as exc:
        failures.append(exc)
    if not failures:
        return
    if not suppress_errors:
        raise failures[0]
    for failure in failures:
        warnings.warn(
            f"suppressed SONIC cleanup failure while preserving the training error: {failure}",
            RuntimeWarning,
            stacklevel=2,
        )


def _validate_checkpoint_token_info(
    checkpoint_tokens: Any, expected_tokens: Mapping[str, Any]
) -> None:
    if not isinstance(checkpoint_tokens, Mapping):
        raise ValueError("SONIC checkpoint token_info must be a mapping")
    required = ("token_dim", "total_dim", "num_tokens", "num_levels", "level_list")
    missing = [key for key in required if key not in checkpoint_tokens]
    if missing:
        raise ValueError(f"SONIC checkpoint token_info is missing fields: {missing}")
    mismatches = {
        key: (checkpoint_tokens[key], expected_tokens[key])
        for key in required
        if checkpoint_tokens[key] != expected_tokens[key]
    }
    if mismatches:
        raise ValueError(f"SONIC checkpoint token contract mismatch: {mismatches}")


def _resolve_sonic_device(
    config: Mapping[str, Any],
    *,
    local_rank: int,
    world_size: int,
    cuda_available: bool,
) -> str:
    configured_device = _get(config, "training.device") or _get(config, "device")
    devices = _get(config, "training.devices")
    configured_devices = tuple(int(value) for value in devices) if devices else None
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if local_rank < 0 or local_rank >= world_size:
        raise ValueError(f"local_rank={local_rank} is out of range for world_size={world_size}")
    if configured_device not in (None, "", "null") and configured_devices is not None:
        raise ValueError("Set either training.device or training.devices, not both")
    if world_size > 1:
        if configured_device not in (None, "", "null"):
            raise ValueError(
                "training.device cannot select one device in a distributed run; "
                "use training.devices"
            )
        if configured_devices is not None and len(configured_devices) != world_size:
            raise ValueError(
                f"training.devices has {len(configured_devices)} entries but "
                f"world_size={world_size}"
            )
    elif configured_devices is not None and len(configured_devices) != 1:
        raise ValueError("a single-process run requires exactly one training.devices entry")
    if not cuda_available:
        return str(configured_device or "cpu")
    if world_size > 1:
        return f"cuda:{local_rank}"
    if configured_devices is not None:
        return f"cuda:{configured_devices[0]}"
    return str(configured_device or "cuda:0")


class SonicPPORunner:
    def __init__(
        self,
        env: Any,
        config: Mapping[str, Any],
        *,
        device: str | torch.device = "cpu",
        log_dir: str | Path | None = None,
    ) -> None:
        if isinstance(config, DictConfig):
            converted = OmegaConf.to_container(config, resolve=True)
            config = cast(Mapping[str, Any], converted)
        if not isinstance(config, Mapping):
            raise TypeError("SONIC runner config must be a mapping")
        self.env, self.config = env, dict(config)
        self.device = torch.device(device)
        dimensions = _get(self.config, "sonic.dimensions", {})
        dimensions = dimensions if isinstance(dimensions, Mapping) else {}
        model_config = _get(self.config, "sonic.model", {})
        model_config = model_config if isinstance(model_config, Mapping) else {}
        self.num_envs = int(
            getattr(
                env,
                "num_envs",
                _get(self.config, "algo.num_envs", _get(self.config, "num_envs", 1)),
            )
        )
        self.horizon = int(
            _get(
                self.config,
                "algo.num_steps_per_env",
                _get(self.config, "num_steps_per_env", _get(self.config, "horizon", 24)),
            )
        )
        self.model = SonicActorCritic(
            actor_obs_dim=int(
                _get(self.config, "actor_obs_dim", dimensions.get("actor_obs_dim", 930))
            ),
            critic_obs_dim=int(
                _get(self.config, "critic_obs_dim", dimensions.get("critic_obs_dim", 1645))
            ),
            tokenizer_obs_dim=int(
                _get(self.config, "tokenizer_obs_dim", dimensions.get("tokenizer_obs_dim", 1761))
            ),
            action_dim=int(_get(self.config, "action_dim", dimensions.get("action_dim", 29))),
            hidden_dims=model_config.get("hidden_dims"),
            actor_hidden_dims=model_config.get("actor_hidden_dims"),
            critic_hidden_dims=model_config.get("critic_hidden_dims"),
            tokenizer_hidden_dim=int(model_config.get("tokenizer_hidden_dim", 512)),
            token_levels=int(model_config.get("token_levels", 32)),
            token_count=int(model_config.get("token_count", 2)),
            critic_obs_normalization=bool(model_config.get("critic_obs_normalization", False)),
            init_noise_std=float(model_config.get("init_noise_std", 0.05)),
            std_clamp_min=float(model_config.get("std_clamp_min", 0.001)),
            std_clamp_max=float(model_config.get("std_clamp_max", 0.5)),
        ).to(self.device)
        _broadcast_model_state(self.model)
        self.algorithm = SonicPPO(self.model, self.config, self.device)
        self.storage = SonicRolloutStorage(
            self.horizon,
            self.num_envs,
            self.model.actor_obs_dim,
            self.model.critic_obs_dim,
            self.model.tokenizer_obs_dim,
            self.model.action_dim,
            self.device,
        )
        self.current_learning_iteration = 0
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.is_main_process = (
            not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
        )
        self.logger = self._build_logger()
        self._logger_finished = False

    def _build_logger(self) -> Any | None:
        backend = _get(self.config, "training.logger")
        if not self.is_main_process or self.log_dir is None or not backend:
            return None
        from unilab.logging.onpolicy import OnPolicyLogger

        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        logger = OnPolicyLogger(
            algo_name="SONIC PPO",
            max_iterations=int(_get(self.config, "algo.max_iterations", 1)),
            num_envs=self.num_envs * world_size,
            num_steps=self.horizon,
            env_name=str(_get(self.config, "training.task_name", "SonicG1Tracking")),
            log_dir=str(self.log_dir),
            log_backend=str(backend),
            wandb_project=str(_get(self.config, "training.wandb_project", "unilab")),
            wandb_entity=_get(self.config, "training.wandb_entity"),
            wandb_name=str(_get(self.config, "training.wandb_name", "")),
            wandb_group=_get(self.config, "training.wandb_group"),
            wandb_job_type=_get(self.config, "training.wandb_job_type"),
            wandb_tags=_get(self.config, "training.wandb_tags"),
            wandb_notes=_get(self.config, "training.wandb_notes"),
        )
        logger.start(status="native sequence PPO")
        return logger

    def _finish_logger(self, *, suppress_errors: bool) -> None:
        if self.logger is None or self._logger_finished:
            return
        self._logger_finished = True
        try:
            self.logger.finish(title="SONIC PPO Training Summary")
        except BaseException as exc:
            if not suppress_errors:
                raise
            warnings.warn(
                f"suppressed SONIC logger cleanup failure while preserving training error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    def _iteration_performance_metrics(
        self,
        *,
        collect_time: float,
        train_time: float,
    ) -> dict[str, float]:
        timing = torch.tensor(
            [collect_time, train_time],
            dtype=torch.float64,
            device=self.device,
        )
        timing_max = timing.clone()
        timing_min = timing.clone()
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(timing_max, op=dist.ReduceOp.MAX)
            dist.all_reduce(timing_min, op=dist.ReduceOp.MIN)
        global_samples = self.horizon * self.num_envs * world_size
        metrics = {
            "perf/rollout_fps": global_samples / max(float(timing_max[0]), 1.0e-9),
            "perf/learner_fps": global_samples / max(float(timing_max[1]), 1.0e-9),
            "perf/iteration_fps": global_samples / max(float(timing_max.sum()), 1.0e-9),
            "perf/collect_rank_skew": float(timing_max[0] / timing_min[0].clamp_min(1.0e-9)),
            "perf/train_rank_skew": float(timing_max[1] / timing_min[1].clamp_min(1.0e-9)),
            "policy/learning_rate": float(self.algorithm.optimizer.param_groups[0]["lr"]),
            "policy/noise_std": float(self.model.std.detach().mean()),
            "policy/optimizer_steps": float(self.algorithm.last_optimizer_steps),
        }
        if self.device.type == "cuda":
            memory = torch.tensor(
                [
                    torch.cuda.memory_allocated(self.device),
                    torch.cuda.max_memory_allocated(self.device),
                ],
                dtype=torch.float64,
                device=self.device,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            metrics["perf/memory_allocated_gib"] = float(memory[0] / 2**30)
            metrics["perf/max_peak_memory_gib"] = float(memory[1] / 2**30)
        return metrics

    def _parse_step_result(
        self, result: Any
    ) -> tuple[Any, Any, torch.Tensor, torch.Tensor, Any | None]:
        """Normalize UniLab/Gym step results and retain terminal observations.

        ``NpEnvState`` keeps the post-autoreset observation in ``obs`` and the
        pre-reset observation in ``final_observation``.  Dropping the latter
        would make timeout transitions bootstrap from a new episode.  The
        tuple branches retain compatibility with both Gymnasium (5-tuple) and
        legacy Gym (4-tuple) environments.
        """

        final_observation: Any | None = None
        info: Any | None = None
        if hasattr(result, "obs"):
            next_obs = result.obs
            rewards = result.reward
            terminated = getattr(result, "terminated", False)
            truncated = getattr(result, "truncated", False)
            final_observation = getattr(result, "final_observation", None)
            info = getattr(result, "info", None)
        else:
            if not isinstance(result, (tuple, list)) or len(result) < 3:
                raise ValueError(
                    "SONIC env.step must return NpEnvState or a tuple with at least "
                    "(obs, reward, done)"
                )
            next_obs, rewards = result[0], result[1]
            if len(result) >= 5:
                terminated, truncated, info = result[2], result[3], result[4]
            elif len(result) == 4:
                # Legacy Gym: (obs, reward, done, info).  A non-mapping fourth
                # value is accepted as a truncated mask for small custom envs.
                terminated = result[2]
                if isinstance(result[3], Mapping):
                    truncated, info = False, result[3]
                else:
                    truncated, info = result[3], None
            else:
                terminated, truncated = result[2], False

        if final_observation is None and isinstance(info, Mapping):
            final_observation = info.get("final_observation")
        terminated_t = _batch_bool(
            terminated, name="terminated", device=self.device, batch=self.num_envs
        )
        truncated_t = _batch_bool(
            truncated, name="truncated", device=self.device, batch=self.num_envs
        )
        return next_obs, rewards, terminated_t, truncated_t, final_observation

    def _timeout_reward_correction(
        self,
        final_observation: Any | None,
        truncated: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``gamma * V(final_observation)`` for timeout rows.

        UniLab autoreset returns reset observations to the policy while
        preserving terminal observations separately.  PPO treats a timeout as
        bootstrap-able, so the terminal value must be added to that transition
        reward before GAE.  If an older/custom environment does not expose a
        terminal observation, the current transition value is the documented
        conservative fallback used by the other UniLab PPO adapters.
        """

        timeout_mask = truncated.to(self.device).bool().reshape(-1)
        if timeout_mask.shape != (self.num_envs,):
            raise ValueError(
                f"truncated mask expects {self.num_envs} values, got {tuple(timeout_mask.shape)}"
            )
        values_flat = values.reshape(-1)
        if values_flat.shape != (self.num_envs,):
            raise ValueError(
                f"value batch expects {self.num_envs} values, got {tuple(values_flat.shape)}"
            )
        correction = torch.zeros_like(values_flat)
        if not bool(timeout_mask.any()):
            return correction

        # Missing terminal observations are possible for legacy/custom envs;
        # preserve their old behavior while making the fallback explicit.
        if final_observation is None:
            correction[timeout_mask] = self.algorithm.gamma * values_flat.detach()[timeout_mask]
            return correction

        try:
            final_actor = _obs(
                final_observation,
                "actor",
                self.model.actor_obs_dim,
                self.device,
                self.num_envs,
            )
            final_critic = _obs(
                final_observation,
                "critic",
                self.model.critic_obs_dim,
                self.device,
                self.num_envs,
            )
            final_token = _obs(
                final_observation,
                "tokenizer",
                self.model.tokenizer_obs_dim,
                self.device,
                self.num_envs,
            )
            with torch.no_grad():
                _, final_values = self.model.distribution(final_actor, final_critic, final_token)
            correction[timeout_mask] = self.algorithm.gamma * final_values.detach()[timeout_mask]
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                "SONIC timeout transition supplied an invalid final_observation; "
                "expected actor/critic/tokenizer observation groups with the configured dimensions"
            ) from exc
        return correction

    def _reset(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hasattr(self.env, "reset"):
            result = self.env.reset()
        elif hasattr(self.env, "init_state"):
            result = self.env.init_state()
        else:
            raise TypeError("SONIC env must provide reset() or init_state()")
        if hasattr(result, "obs"):
            result = result.obs
        result = result[0] if isinstance(result, tuple) else result
        return (
            _obs(result, "actor", self.model.actor_obs_dim, self.device, self.num_envs),
            _obs(result, "critic", self.model.critic_obs_dim, self.device, self.num_envs),
            _obs(result, "tokenizer", self.model.tokenizer_obs_dim, self.device, self.num_envs),
        )

    def learn(self, num_learning_iterations: int) -> dict[str, float]:
        if num_learning_iterations < 0:
            raise ValueError("num_learning_iterations must be non-negative")
        learning_error: BaseException | None = None
        try:
            if num_learning_iterations == 0:
                return {}
            self.model.eval()
            actor_obs, critic_obs, token_obs = self._reset()
            metrics: dict[str, float] = {}
            for iteration in range(
                self.current_learning_iteration,
                self.current_learning_iteration + int(num_learning_iterations),
            ):
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                self.model.eval()
                _synchronize_cuda(self.device)
                collect_started = time.perf_counter()
                reward_sum = torch.zeros((), dtype=torch.float64, device=self.device)
                with torch.no_grad():
                    for _ in range(self.horizon):
                        rollout_actor_obs = _rollout_snapshot(actor_obs)
                        rollout_critic_obs = _rollout_snapshot(critic_obs)
                        rollout_token_obs = _rollout_snapshot(token_obs)
                        distribution, values = self.model.distribution(
                            rollout_actor_obs, rollout_critic_obs, rollout_token_obs
                        )
                        actions = distribution.sample()
                        log_probs = distribution.log_prob(actions).sum(-1)
                        result = self.env.step(actions.detach().cpu().numpy())
                        next_obs, rewards, terminated, truncated, final_observation = (
                            self._parse_step_result(result)
                        )
                        dones = terminated | truncated
                        rewards_t = torch.as_tensor(
                            rewards, device=self.device, dtype=torch.float32
                        ).reshape(-1)
                        if rewards_t.shape != (self.num_envs,):
                            raise ValueError(
                                f"SONIC rewards expect {self.num_envs} values, "
                                f"got {tuple(rewards_t.shape)}"
                            )
                        rewards_t = rewards_t + self._timeout_reward_correction(
                            final_observation,
                            truncated & ~terminated,
                            values,
                        )
                        reward_sum += rewards_t.sum(dtype=torch.float64)
                        self.storage.add(
                            rollout_actor_obs,
                            rollout_critic_obs,
                            rollout_token_obs,
                            actions,
                            rewards_t,
                            dones,
                            values,
                            log_probs,
                            distribution.mean,
                            distribution.stddev,
                        )
                        actor_obs, critic_obs, token_obs = (
                            _obs(
                                next_obs,
                                "actor",
                                self.model.actor_obs_dim,
                                self.device,
                                self.num_envs,
                            ),
                            _obs(
                                next_obs,
                                "critic",
                                self.model.critic_obs_dim,
                                self.device,
                                self.num_envs,
                            ),
                            _obs(
                                next_obs,
                                "tokenizer",
                                self.model.tokenizer_obs_dim,
                                self.device,
                                self.num_envs,
                            ),
                        )
                _synchronize_cuda(self.device)
                collect_time = time.perf_counter() - collect_started
                train_started = time.perf_counter()
                with torch.no_grad():
                    _, last_values = self.model.distribution(actor_obs, critic_obs, token_obs)
                self.storage.compute_returns(last_values, self.algorithm.gamma, self.algorithm.lam)
                metrics = self.algorithm.update(self.storage)
                # Keep one immutable normalizer snapshot across collection and PPO.
                # The just-consumed rollout updates the statistics for the next
                # iteration, then rank-local moments are merged exactly once.
                self.model.begin_normalizer_update()
                self.model.update_normalizers(self.storage.critic_obs)
                self.model.synchronize_normalizers()
                _synchronize_cuda(self.device)
                train_time = time.perf_counter() - train_started
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(reward_sum, op=dist.ReduceOp.SUM)
                    reward_denominator = self.horizon * self.num_envs * dist.get_world_size()
                else:
                    reward_denominator = self.horizon * self.num_envs
                metrics.update(
                    self._iteration_performance_metrics(
                        collect_time=collect_time,
                        train_time=train_time,
                    )
                )
                self.current_learning_iteration = iteration + 1
                if self.logger is not None:
                    self.logger.log_step(
                        self.current_learning_iteration,
                        metrics,
                        reward=float(reward_sum / max(1, reward_denominator)),
                        collect_time=collect_time,
                        train_time=train_time,
                    )
                save_interval = int(
                    _get(self.config, "algo.save_interval", _get(self.config, "save_interval", 500))
                )
                if (
                    self.log_dir is not None
                    and self.is_main_process
                    and save_interval > 0
                    and self.current_learning_iteration % save_interval == 0
                ):
                    self.save(self.log_dir / f"model_{self.current_learning_iteration}.pt")
            if self.log_dir is not None and self.is_main_process:
                self.save(self.log_dir / "last.pt")
            return metrics
        except BaseException as exc:
            learning_error = exc
            raise
        finally:
            self._finish_logger(suppress_errors=learning_error is not None)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        token_info = self.model.tokenizer.get_token_info()
        algorithm_state = {
            key: value for key, value in self.algorithm.state_dict().items() if key != "optimizer"
        }
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.algorithm.optimizer.state_dict(),
            "algorithm": algorithm_state,
            "iteration": self.current_learning_iteration,
            "token_info": token_info,
            "contract": {
                "actor_obs_dim": self.model.actor_obs_dim,
                "critic_obs_dim": self.model.critic_obs_dim,
                "tokenizer_obs_dim": self.model.tokenizer_obs_dim,
                "action_dim": self.model.action_dim,
                "horizon": self.horizon,
                "token_info": token_info,
            },
        }
        # A rank-0 write can be observed by a resume process while the file is
        # still being serialized.  Replace atomically on the same filesystem
        # so readers see either the previous complete checkpoint or this one.
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            torch.save(state, temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _load_checkpoint_state(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        if (
            not isinstance(state, Mapping)
            or "model" not in state
            or not isinstance(state["model"], Mapping)
        ):
            raise ValueError("SONIC checkpoint must contain a model state mapping")
        has_contract = "contract" in state
        if has_contract:
            contract = state["contract"]
            if not isinstance(contract, Mapping):
                raise ValueError("SONIC checkpoint contract must be a mapping")
            expected_contract = {
                "actor_obs_dim": self.model.actor_obs_dim,
                "critic_obs_dim": self.model.critic_obs_dim,
                "tokenizer_obs_dim": self.model.tokenizer_obs_dim,
                "action_dim": self.model.action_dim,
                "horizon": self.horizon,
            }
            required_contract = (*expected_contract, "token_info")
            missing = [key for key in required_contract if key not in contract]
            if missing:
                raise ValueError(f"SONIC checkpoint contract is missing fields: {missing}")
            mismatches = {
                key: (contract[key], value)
                for key, value in expected_contract.items()
                if contract[key] != value
            }
            if mismatches:
                raise ValueError(f"SONIC checkpoint contract mismatch: {mismatches}")
            _validate_checkpoint_token_info(
                contract["token_info"], self.model.tokenizer.get_token_info()
            )
        model_state = dict(state["model"])
        # Contract-less checkpoints are legacy model-only warm starts. The
        # earlier native prototype used log_std before adopting direct std.
        if not has_contract and "std" not in model_state and "log_std" in model_state:
            model_state["std"] = torch.exp(torch.as_tensor(model_state.pop("log_std")))
        try:
            self.model.load_state_dict(model_state)
        except RuntimeError as exc:
            raise ValueError(f"SONIC checkpoint model shape mismatch: {exc}") from exc
        if not has_contract:
            return

        algorithm_state = state.get("algorithm")
        if algorithm_state is not None and not isinstance(algorithm_state, Mapping):
            raise ValueError("SONIC checkpoint algorithm state must be a mapping")
        optimizer_state = state.get("optimizer")
        if "optimizer" not in state and isinstance(algorithm_state, Mapping):
            optimizer_state = algorithm_state.get("optimizer")
        if optimizer_state is not None:
            self.algorithm.optimizer.load_state_dict(optimizer_state)
        if isinstance(algorithm_state, Mapping):
            self.algorithm.load_state_dict(
                {key: value for key, value in algorithm_state.items() if key != "optimizer"}
            )
        self.current_learning_iteration = int(state.get("iteration", 0))

    def load(self, path: str | Path) -> None:
        resolved_path = Path(path).expanduser().resolve()
        _validate_distributed_checkpoint(resolved_path)
        load_error: BaseException | None = None
        try:
            self._load_checkpoint_state(resolved_path)
        except BaseException as exc:
            load_error = exc
        try:
            _synchronize_checkpoint_load(
                error=load_error,
                iteration=None if load_error is not None else self.current_learning_iteration,
            )
        except BaseException as sync_error:
            if load_error is not None:
                raise sync_error from load_error
            raise
        if load_error is not None:
            raise load_error


def train_sonic(
    cfg: Any, plan: Any = None, env_cfg_override: Mapping[str, Any] | None = None, env: Any = None
) -> SonicPPORunner:
    """Build and run SONIC PPO; ``env``/``env_factory`` are injection points for tests."""
    if isinstance(cfg, DictConfig):
        config = OmegaConf.to_container(cfg, resolve=True)
    else:
        config = cfg if isinstance(cfg, Mapping) else dict(cfg)
    if not isinstance(config, Mapping):
        raise TypeError("SONIC config must resolve to a mapping")
    try:
        validate_sonic_owner(config)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    created_process_group = False
    resolved_env: Any = None
    runner: SonicPPORunner | None = None
    training_error: BaseException | None = None
    try:
        resources = getattr(plan, "resources", None)
        if resources is not None:
            apply_sonic_torch_threads(resources, torch_runtime=torch)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        device = _resolve_sonic_device(
            config,
            local_rank=local_rank,
            world_size=world_size,
            cuda_available=torch.cuda.is_available(),
        )
        if str(device).startswith("cuda") and torch.cuda.is_available():
            # NCCL captures the current CUDA device while the process group is
            # initialized; select the rank-local device before creating it.
            torch.cuda.set_device(torch.device(device))
        distributed_env = all(name in os.environ for name in ("MASTER_ADDR", "MASTER_PORT", "RANK"))
        if dist.is_available() and not dist.is_initialized() and distributed_env and world_size > 1:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            try:
                dist.init_process_group(backend=backend, init_method="env://")
            finally:
                created_process_group = dist.is_initialized()

        resolved_env = env if env is not None else _get(config, "env_instance")
        factory = _get(config, "env_factory")
        if resolved_env is None and callable(factory):
            kwargs = {"cfg": config, "env_cfg_override": dict(env_cfg_override or {})}
            resolved_env = (
                factory(**kwargs)
                if "env_cfg_override" in inspect.signature(factory).parameters
                else factory(config)
            )
        if resolved_env is None:
            # Keep registry construction on the cold path.  This is the production
            # route used by torchrun when no test env/factory is injected.
            from unilab.base import registry

            registry.ensure_registries()
            task_name = str(_get(config, "training.task_name", "SonicG1Tracking"))
            sim_backend = str(_get(config, "training.sim_backend", "mujoco"))
            num_envs = int(
                getattr(plan.report, "num_envs_per_rank", 0)
                if plan is not None and getattr(plan, "report", None) is not None
                else _get(config, "algo.num_envs", _get(config, "num_envs", 1))
            )
            if num_envs < 1:
                raise ValueError(f"invalid SONIC num_envs={num_envs}")
            resolved_env = registry.make(
                task_name,
                sim_backend=sim_backend,
                env_cfg_override=dict(env_cfg_override or {}),
                num_envs=num_envs,
            )

        log_dir = getattr(plan, "log_dir", _get(config, "training.log_dir"))
        runner = SonicPPORunner(resolved_env, config, device=device, log_dir=log_dir)
        resume = _get(config, "resume") or _get(config, "training.resume")
        if resume:
            runner.load(resume)
        target_iterations = int(
            _get(config, "max_iterations", _get(config, "algo.max_iterations", 1))
        )
        runner.learn(max(0, target_iterations - runner.current_learning_iteration))
        return runner
    except BaseException as exc:
        training_error = exc
        raise
    finally:
        _cleanup_sonic_runtime(
            runner=runner,
            env=resolved_env,
            owned_process_group=created_process_group,
            suppress_errors=training_error is not None,
        )
