from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from unilab.algos.torch.sonic_ppo import (
    FSQ,
    SonicActorCritic,
    SonicPPO,
    SonicPPORunner,
    SonicRolloutStorage,
    train_sonic,
)
from unilab.algos.torch.sonic_ppo.runner import (
    _broadcast_model_state,
    _cleanup_sonic_runtime,
    _finish_sonic_distributed,
    _resolve_sonic_device,
    _synchronize_checkpoint_load,
    _synchronize_cuda,
    _validate_distributed_checkpoint,
)
from unilab.base.np_env import NpEnvState


def _obs(num_envs: int) -> dict[str, torch.Tensor]:
    return {
        "policy": torch.zeros(num_envs, 930),
        "privileged": torch.zeros(num_envs, 1645),
        "tokens": torch.zeros(num_envs, 1761),
    }


@pytest.fixture(scope="module")
def sonic_v11_specs() -> dict[str, object]:
    config_path = Path(__file__).resolve().parents[2] / "conf/sonic/config.yaml"
    config = yaml.safe_load(config_path.read_text())
    return config["sonic"]["model"]


def _named_model(specs: dict[str, object]) -> SonicActorCritic:
    return SonicActorCritic(
        hidden_dims=(8,),
        model_profile="sonic_v1_1",
        tokenizer_fields=specs["tokenizer_fields"],
        encoders=specs["encoders"],
        decoders=specs["decoders"],
        token_count=specs["token_count"],
        token_levels=specs["token_levels"],
        critic_hidden_dims=(8,),
    )


def _named_token_obs(route: tuple[int, ...], batch_size: int = 1) -> torch.Tensor:
    observations = torch.zeros(batch_size, 1761)
    observations[:, : len(route)] = torch.tensor(route, dtype=observations.dtype)
    return observations


def _named_runner_config(specs: dict[str, object]) -> dict[str, object]:
    return {
        "num_steps_per_env": 1,
        "num_mini_batches": 1,
        "num_learning_epochs": 1,
        "dimensions": {"actor_obs_dim": 930, "critic_obs_dim": 1645, "tokenizer_obs_dim": 1761, "action_dim": 29},
        "sonic": {"model": {**specs, "profile": "sonic_v1_1", "hidden_dims": [8], "critic_hidden_dims": [8]}},
    }


def test_named_model_reads_config_and_routes_one_hot_gradients(sonic_v11_specs) -> None:
    model = _named_model(sonic_v11_specs)
    for route_index, encoder_name in enumerate(("g1", "teleop", "smpl")):
        model.zero_grad()
        tokens = model.tokenizer(_named_token_obs(tuple(int(index == route_index) for index in range(3))))
        tokens.sum().backward()
        for name, encoder in model.tokenizer.encoders.items():
            gradients = [parameter.grad for parameter in encoder.parameters()]
            has_gradient = any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients)
            assert has_gradient is (name == encoder_name)


def test_named_model_multihot_g1_smpl_uses_smpl_token(sonic_v11_specs) -> None:
    model = _named_model(sonic_v11_specs)
    smpl_tokens = model.tokenizer(_named_token_obs((0, 0, 1)))
    paired_tokens = model.tokenizer(_named_token_obs((1, 0, 1)))
    assert torch.equal(paired_tokens, smpl_tokens)


def test_named_outputs_shapes_and_both_decoders_receive_gradients(sonic_v11_specs) -> None:
    model = _named_model(sonic_v11_specs)
    output = model.named_outputs(torch.zeros(3, 930), _named_token_obs((1, 0, 0), 3))
    decoded = output["decoded_outputs"]
    assert output["action_mean"].shape == (3, 29)
    assert decoded["g1_dyn"]["action"].shape == (3, 29)
    assert decoded["g1_kin"]["command_multi_future_nonflat"].shape == (3, 10, 58)
    assert decoded["g1_kin"]["motion_anchor_ori_heading_mf_nonflat"].shape == (3, 10, 6)
    sum(decoded_name[output_name].sum() for decoded_name in decoded.values() for output_name in decoded_name).backward()
    for decoder in model.tokenizer.decoders.values():
        assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in decoder.parameters())


def test_named_auxiliary_losses_route_masks_and_pair_mse(sonic_v11_specs) -> None:
    model = _named_model(sonic_v11_specs)
    routes = ((1, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 1))
    token_obs = torch.cat([_named_token_obs(route) for route in routes])
    outputs = model.tokenizer.decode(token_obs, torch.zeros(4, 930))
    losses = model.tokenizer.auxiliary_losses(outputs)

    assert set(losses) == {
        "g1_recon",
        "g1_smpl_latent",
        "g1_teleop_latent",
        "teleop_smpl_latent",
        "reencoded_smpl_g1_latent",
    }
    assert all(torch.isfinite(loss) for loss in losses.values())
    latents = outputs["encoded_latents"]
    masks = outputs["encoder_masks"]
    assert losses["g1_smpl_latent"] == torch.nn.functional.mse_loss(
        latents["g1"][masks["g1_has_smpl"]], latents["smpl"]
    )
    assert losses["g1_teleop_latent"] == torch.nn.functional.mse_loss(
        latents["g1"][masks["g1_has_teleop"]], latents["teleop"][masks["teleop_has_g1"]]
    )
    assert losses["teleop_smpl_latent"] == torch.nn.functional.mse_loss(
        latents["teleop"][masks["teleop_has_smpl"]], latents["smpl"][masks["smpl_has_teleop"]]
    )


def test_named_auxiliary_losses_backward_reaches_all_encoders_and_decoders(sonic_v11_specs) -> None:
    model = _named_model(sonic_v11_specs)
    routes = ((1, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 1))
    outputs = model.tokenizer.decode(
        torch.cat([_named_token_obs(route) for route in routes]), torch.zeros(4, 930)
    )
    sum(model.tokenizer.auxiliary_losses(outputs).values()).backward()
    for module in model.tokenizer.encoders.values():
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in module.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.tokenizer.decoders["g1_kin"].parameters()
    )


def test_named_auxiliary_losses_sparse_pairs_are_connected_zero(sonic_v11_specs) -> None:
    model = _named_model(sonic_v11_specs)
    outputs = model.tokenizer.decode(_named_token_obs((1, 0, 0)), torch.zeros(1, 930))
    losses = model.tokenizer.auxiliary_losses(outputs)
    pair_names = {
        "g1_smpl_latent",
        "g1_teleop_latent",
        "teleop_smpl_latent",
        "reencoded_smpl_g1_latent",
    }
    assert all(losses[name].item() == 0 for name in pair_names)
    assert all(losses[name].grad_fn is not None for name in pair_names)
    sum(losses.values()).backward()


def test_named_ppo_update_uses_one_training_forward_per_microbatch(sonic_v11_specs) -> None:
    torch.manual_seed(0)
    model = _named_model(sonic_v11_specs)
    storage = SonicRolloutStorage(1, 4)
    actor = torch.randn(4, 930)
    critic = torch.randn(4, 1645)
    tokenizer = torch.cat([_named_token_obs(route) for route in ((1, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 1))])
    distribution, values = model.distribution(actor, critic, tokenizer)
    actions = distribution.sample()
    storage.add(actor, critic, tokenizer, actions, torch.arange(1, 5, dtype=torch.float32), torch.zeros(4), values,
                distribution.log_prob(actions).sum(-1), distribution.mean, distribution.stddev)
    storage.compute_returns(torch.zeros(4))
    algorithm = SonicPPO(
        model,
        num_learning_epochs=1,
        num_mini_batches=1,
        microbatch_size=1,
        aux_loss_coef={name: 1.0 for name in (
            "g1_recon", "g1_smpl_latent", "g1_teleop_latent", "teleop_smpl_latent",
            "reencoded_smpl_g1_latent",
        )},
    )
    policy, _, auxiliary = model.training_forward(actor, critic, tokenizer)
    (policy.mean.square().mean() + sum(auxiliary.values())).backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.tokenizer.decoders["g1_dyn"].parameters()
    )
    model.zero_grad(set_to_none=True)
    calls = 0
    original_forward = model.training_forward

    def training_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(*args, **kwargs)

    model.training_forward = training_forward
    model.distribution = lambda *args, **kwargs: pytest.fail("distribution called during update")
    model.auxiliary_losses = lambda *args, **kwargs: pytest.fail("auxiliary_losses called during update")
    metrics = algorithm.update(storage)
    assert calls == 4
    assert {
        "g1_recon", "g1_smpl_latent", "g1_teleop_latent", "teleop_smpl_latent",
        "reencoded_smpl_g1_latent",
    } <= set(metrics)
    assert all(np.isfinite(metrics[name]) for name in metrics)


@pytest.mark.parametrize("version", [None, "unilab_sonic_dense_test.v1"])
def test_named_runner_rejects_missing_or_prototype_contract_version(tmp_path, sonic_v11_specs, version) -> None:
    runner = SonicPPORunner(_StateEnv(), _named_runner_config(sonic_v11_specs), device="cpu")
    checkpoint = tmp_path / "checkpoint.pt"
    runner.save(checkpoint)
    state = torch.load(checkpoint, weights_only=False)
    if version is None:
        del state["contract"]["model_contract_version"]
    else:
        state["contract"]["model_contract_version"] = version
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="contract/version mismatch"):
        runner.load(checkpoint)


def test_named_model_supports_tiny_native_ppo_update(sonic_v11_specs) -> None:
    torch.manual_seed(0)
    model = _named_model(sonic_v11_specs)
    storage = SonicRolloutStorage(1, 1)
    actor = torch.randn(1, 930)
    critic = torch.randn(1, 1645)
    tokenizer = _named_token_obs((1, 0, 0))
    distribution, value = model.distribution(actor, critic, tokenizer)
    actions = distribution.sample()
    storage.add(actor, critic, tokenizer, actions, torch.zeros(1), torch.zeros(1), value, distribution.log_prob(actions).sum(-1), distribution.mean, distribution.stddev)
    storage.compute_returns(torch.zeros(1))
    algorithm = SonicPPO(model, num_learning_epochs=1, num_mini_batches=1)
    metrics = algorithm.update(storage)
    assert set(metrics) == {"loss", "value_loss", "policy_loss", "entropy", "approx_kl"}
    assert algorithm.last_optimizer_steps == 1


def test_fsq_contract_shape_and_range() -> None:
    quantizer = FSQ()
    values = quantizer(torch.randn(3, 2))
    assert values.shape == (3, 2)
    assert float(values.min()) >= -1.0
    assert float(values.max()) <= 15.0 / 16.0
    indices = quantizer.indices(values)
    assert int(indices.min()) >= 0 and int(indices.max()) <= 31


def test_actor_uses_release_direct_clamped_std() -> None:
    model = SonicActorCritic(hidden_dims=(8,), tokenizer_hidden_dim=8)
    assert "std" in dict(model.named_parameters())
    assert "log_std" not in dict(model.named_parameters())
    with torch.no_grad():
        model.std.fill_(10.0)
    distribution, _ = model.distribution(*_obs(2).values())
    assert torch.allclose(distribution.stddev, torch.full((2, 29), 0.5))


def test_runner_accepts_tokenizer_obs_group_alias() -> None:
    from unilab.algos.torch.sonic_ppo.runner import _obs as resolve_obs

    value = resolve_obs(
        {"tokenizer_obs": torch.zeros(2, 1761)},
        "tokenizer",
        1761,
        torch.device("cpu"),
        2,
    )
    assert value.shape == (2, 1761)


def test_native_ppo_defaults_to_release_value_coefficient() -> None:
    model = SonicActorCritic(hidden_dims=(8,), tokenizer_hidden_dim=8)
    assert SonicPPO(model).value_loss_coef == pytest.approx(1.0)


def test_critic_does_not_depend_on_tokenizer_input() -> None:
    model = SonicActorCritic(hidden_dims=(8,), tokenizer_hidden_dim=8)
    actor = torch.zeros(2, 930)
    critic = torch.zeros(2, 1645)
    token_a = torch.zeros(2, 1761)
    token_b = torch.ones(2, 1761)
    _, value_a = model.distribution(actor, critic, token_a)
    _, value_b = model.distribution(actor, critic, token_b)
    assert torch.allclose(value_a, value_b)


def test_sonic_ppo_five_by_four_update() -> None:
    model = SonicActorCritic(hidden_dims=(16, 8), tokenizer_hidden_dim=8)
    storage = SonicRolloutStorage(2, 8)
    for _ in range(2):
        actor, critic, tokenizer = torch.randn(8, 930), torch.randn(8, 1645), torch.randn(8, 1761)
        distribution, values = model.distribution(actor, critic, tokenizer)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions).sum(-1)
        storage.add(
            actor,
            critic,
            tokenizer,
            actions,
            torch.randn(8),
            torch.zeros(8),
            values,
            log_probs,
            distribution.mean,
            distribution.stddev,
        )
    storage.compute_returns(torch.zeros(8))
    algorithm = SonicPPO(model, num_learning_epochs=5, num_mini_batches=4, microbatch_size=1)
    metrics = algorithm.update(storage)
    assert set(metrics) == {"loss", "value_loss", "policy_loss", "entropy", "approx_kl"}
    assert algorithm.last_optimizer_steps == 40


def test_microbatch_accumulation_preserves_logical_optimizer_steps() -> None:
    model = SonicActorCritic(
        actor_obs_dim=4,
        critic_obs_dim=5,
        tokenizer_obs_dim=6,
        action_dim=2,
        hidden_dims=(8,),
        tokenizer_hidden_dim=8,
    )
    storage = SonicRolloutStorage(2, 8, 4, 5, 6, 2)
    for _ in range(2):
        actor = torch.randn(8, 4)
        critic = torch.randn(8, 5)
        tokenizer = torch.randn(8, 6)
        distribution, values = model.distribution(actor, critic, tokenizer)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions).sum(-1)
        storage.add(
            actor,
            critic,
            tokenizer,
            actions,
            torch.randn(8),
            torch.zeros(8),
            values,
            log_probs,
            distribution.mean,
            distribution.stddev,
        )
    storage.compute_returns(torch.zeros(8))
    algorithm = SonicPPO(
        model,
        num_learning_epochs=5,
        num_mini_batches=4,
        microbatch_size=1,
        optimizer_step_per_microbatch=False,
    )

    algorithm.update(storage)

    assert algorithm.last_optimizer_steps == 20


def test_adaptive_lr_uses_exact_gaussian_kl_per_microbatch(monkeypatch) -> None:
    model = SonicActorCritic(
        actor_obs_dim=2,
        critic_obs_dim=2,
        tokenizer_obs_dim=2,
        action_dim=2,
        hidden_dims=(4,),
        tokenizer_hidden_dim=4,
    )
    storage = SonicRolloutStorage(1, 4, 2, 2, 2, 2)
    actor = torch.zeros(4, 2)
    critic = torch.zeros(4, 2)
    tokenizer = torch.zeros(4, 2)
    old_distribution, values = model.distribution(actor, critic, tokenizer)
    actions = old_distribution.sample()
    storage.add(
        actor,
        critic,
        tokenizer,
        actions,
        torch.zeros(4),
        torch.zeros(4),
        values,
        old_distribution.log_prob(actions).sum(-1),
        old_distribution.mean,
        old_distribution.stddev,
    )
    storage.compute_returns(torch.zeros(4))
    with torch.no_grad():
        model.actor[-1].bias.add_(0.1)
    algorithm = SonicPPO(
        model,
        num_learning_epochs=1,
        num_mini_batches=2,
        microbatch_size=1,
        learning_rate=0.0,
        schedule="adaptive",
        desired_kl=0.01,
    )
    observed_kl: list[float] = []
    monkeypatch.setattr(algorithm, "_adjust_learning_rate", observed_kl.append)

    metrics = algorithm.update(storage)

    assert len(observed_kl) == 4
    assert all(value > 3.9 for value in observed_kl)
    assert metrics["approx_kl"] > 3.9


class _StateEnv:
    num_envs = 2

    def init_state(self) -> NpEnvState:
        return NpEnvState(
            {
                "policy": np.zeros((2, 930)),
                "privileged": np.zeros((2, 1645)),
                "tokens": np.zeros((2, 1761)),
            },
            np.zeros(2),
            np.zeros(2, bool),
            np.zeros(2, bool),
            {},
        )

    def step(self, actions: np.ndarray) -> NpEnvState:
        assert actions.shape == (2, 29)
        return self.init_state()


def test_runner_np_env_state_checkpoint_resume(tmp_path) -> None:
    config = {
        "num_steps_per_env": 1,
        "max_iterations": 1,
        "num_mini_batches": 1,
        "num_learning_epochs": 1,
        "save_interval": 99,
        "algo": {"save_interval": 1},
    }
    runner = SonicPPORunner(_StateEnv(), config, device="cpu", log_dir=tmp_path)
    runner.learn(1)
    checkpoint = tmp_path / "model_1.pt"
    assert checkpoint.is_file()
    assert (tmp_path / "last.pt").is_file()
    assert not list(tmp_path.glob(".model_1.pt.tmp-*"))
    state = torch.load(tmp_path / "last.pt", weights_only=False)
    assert "optimizer" not in state["algorithm"]
    restored = SonicPPORunner(_StateEnv(), config, device="cpu")
    restored.load(checkpoint)
    assert restored.current_learning_iteration == 1


def test_runner_always_saves_last_checkpoint_outside_periodic_interval(tmp_path) -> None:
    class TinyEnv:
        num_envs = 1

        @staticmethod
        def _state() -> dict[str, np.ndarray]:
            return {
                "policy": np.zeros((1, 1), dtype=np.float32),
                "privileged": np.zeros((1, 1), dtype=np.float32),
                "tokens": np.zeros((1, 1), dtype=np.float32),
            }

        def init_state(self) -> dict[str, np.ndarray]:
            return self._state()

        def step(self, actions: np.ndarray):
            assert actions.shape == (1, 1)
            return self._state(), np.zeros(1), np.zeros(1, dtype=bool)

    config = {
        **_checkpoint_config(),
        "save_interval": 99,
        "sonic": {
            "model": {
                "hidden_dims": [8],
                "tokenizer_hidden_dim": 8,
            }
        },
    }
    runner = SonicPPORunner(TinyEnv(), config, device="cpu", log_dir=tmp_path)

    runner.learn(1)

    assert (tmp_path / "last.pt").is_file()
    assert not (tmp_path / "model_1.pt").exists()


def test_cuda_timing_sync_only_runs_for_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    synchronized: list[str] = []
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: synchronized.append(str(device)),
    )

    _synchronize_cuda(torch.device("cpu"))
    _synchronize_cuda(torch.device("cuda:3"))

    assert synchronized == ["cuda:3"]


def test_model_state_broadcast_includes_parameters_and_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))
    model.register_buffer("running", torch.zeros(1))
    broadcast: list[torch.Tensor] = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda tensor, src: broadcast.append(tensor),
    )

    _broadcast_model_state(model)

    assert broadcast == [model.weight, model.running]


def test_distributed_resume_rejects_different_checkpoint_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    checkpoint = tmp_path / "resume.pt"
    checkpoint.write_bytes(b"rank-zero")
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(output, local_result) -> None:
        output[:] = [local_result, {"digest": "different", "error": None}]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    with pytest.raises(ValueError, match="differ across distributed ranks"):
        _validate_distributed_checkpoint(checkpoint)


def test_owned_process_group_is_destroyed_without_cleanup_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle: list[str] = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: lifecycle.append("barrier"))
    monkeypatch.setattr(
        torch.distributed,
        "destroy_process_group",
        lambda: lifecycle.append("destroy"),
    )

    _finish_sonic_distributed(owned=True)

    assert lifecycle == ["destroy"]


def test_distributed_checkpoint_load_failure_is_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(output, local_result) -> None:
        output[:] = [local_result, {"error": "RuntimeError: rank one", "iteration": None}]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    with pytest.raises(ValueError, match="load failed on distributed rank"):
        _synchronize_checkpoint_load(error=None, iteration=4)


def test_cleanup_preserves_primary_error_and_attempts_every_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle: list[str] = []

    class BadEnv:
        @staticmethod
        def close() -> None:
            lifecycle.append("close")
            raise RuntimeError("close failed")

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def destroy() -> None:
        lifecycle.append("destroy")
        raise RuntimeError("destroy failed")

    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroy)

    with pytest.warns(RuntimeWarning, match="suppressed SONIC cleanup failure"):
        _cleanup_sonic_runtime(
            runner=None,
            env=BadEnv(),
            owned_process_group=True,
            suppress_errors=True,
        )

    assert lifecycle == ["close", "destroy"]


def test_critic_rms_freezes_in_eval_and_updates_per_train_batch() -> None:
    config = {
        "num_steps_per_env": 1,
        "num_mini_batches": 1,
        "num_learning_epochs": 1,
        "sonic": {
            "model": {
                "hidden_dims": [8],
                "tokenizer_hidden_dim": 8,
                "critic_obs_normalization": True,
            }
        },
    }
    runner = SonicPPORunner(_StateEnv(), config, device="cpu")
    assert runner.model.critic_rms is not None
    critic_rms = runner.model.critic_rms
    actor_obs = torch.zeros(2, runner.model.actor_obs_dim)
    critic_obs = torch.ones(2, runner.model.critic_obs_dim)

    assert float(critic_rms.count) == pytest.approx(1.0)
    runner.model.eval()
    runner.model.distribution(actor_obs, critic_obs)
    assert float(critic_rms.count) == pytest.approx(1.0)

    runner.model.train()
    runner.model.distribution(actor_obs, critic_obs)
    assert float(critic_rms.count) == pytest.approx(3.0)
    runner.model.distribution(actor_obs, critic_obs)
    assert float(critic_rms.count) == pytest.approx(5.0)


def test_train_resume_runs_only_remaining_target_iterations(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    checkpoint_runner = SonicPPORunner(_StateEnv(), {}, device="cpu")
    checkpoint_runner.current_learning_iteration = 2
    checkpoint = tmp_path / "resume.pt"
    checkpoint_runner.save(checkpoint)
    learned: list[int] = []
    monkeypatch.setattr(
        SonicPPORunner,
        "learn",
        lambda self, iterations: learned.append(int(iterations)) or {},
    )

    train_sonic(
        {
            "algo": {"max_iterations": 5},
            "training": {"resume": str(checkpoint)},
        },
        env=_StateEnv(),
    )

    assert learned == [3]


def test_train_closes_injected_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    class ClosableStateEnv(_StateEnv):
        closed = False

        def close(self) -> None:
            self.closed = True

    env = ClosableStateEnv()
    monkeypatch.setattr(SonicPPORunner, "learn", lambda self, iterations: {})

    train_sonic({"algo": {"max_iterations": 0}}, env=env)

    assert env.closed


def test_train_finishes_logger_when_resume_load_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class Logger:
        finished = False

        def finish(self, **_: object) -> None:
            self.finished = True

    logger = Logger()
    monkeypatch.setattr(SonicPPORunner, "_build_logger", lambda self: logger)

    with pytest.raises(FileNotFoundError):
        train_sonic(
            {
                "training": {"resume": str(tmp_path / "missing.pt")},
                "algo": {"max_iterations": 1},
            },
            env=_StateEnv(),
        )

    assert logger.finished


def test_runner_finishes_logger_when_training_raises() -> None:
    class FailingEnv:
        num_envs = 1

        @staticmethod
        def init_state() -> dict[str, np.ndarray]:
            return {
                "policy": np.zeros((1, 1), dtype=np.float32),
                "privileged": np.zeros((1, 1), dtype=np.float32),
                "tokens": np.zeros((1, 1), dtype=np.float32),
            }

        @staticmethod
        def step(actions: np.ndarray) -> None:
            del actions
            raise RuntimeError("step failed")

    class Logger:
        finished = False

        def finish(self, **_: object) -> None:
            self.finished = True

    runner = SonicPPORunner(FailingEnv(), _checkpoint_config(), device="cpu")
    logger = Logger()
    runner.logger = logger

    with pytest.raises(RuntimeError, match="step failed"):
        runner.learn(1)

    assert logger.finished


def test_logger_cleanup_does_not_hide_training_error() -> None:
    class FailingEnv:
        num_envs = 1

        @staticmethod
        def init_state() -> dict[str, np.ndarray]:
            return {
                "policy": np.zeros((1, 1), dtype=np.float32),
                "privileged": np.zeros((1, 1), dtype=np.float32),
                "tokens": np.zeros((1, 1), dtype=np.float32),
            }

        @staticmethod
        def step(actions: np.ndarray) -> None:
            del actions
            raise RuntimeError("primary step failure")

    class BadLogger:
        @staticmethod
        def finish(**_: object) -> None:
            raise RuntimeError("secondary logger failure")

    runner = SonicPPORunner(FailingEnv(), _checkpoint_config(), device="cpu")
    runner.logger = BadLogger()

    with pytest.warns(RuntimeWarning, match="logger cleanup failure"):
        with pytest.raises(RuntimeError, match="primary step failure"):
            runner.learn(1)


def test_train_applies_torch_intra_and_interop_budget(monkeypatch) -> None:
    applied: dict[str, int] = {}
    monkeypatch.setattr(torch, "set_num_threads", lambda value: applied.update(intra=value))
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 76)
    monkeypatch.setattr(
        torch,
        "set_num_interop_threads",
        lambda value: applied.update(interop=value),
    )
    monkeypatch.setattr(SonicPPORunner, "learn", lambda self, iterations: {})
    plan = SimpleNamespace(
        resources=SimpleNamespace(
            torch_num_threads=2,
            torch_num_interop_threads=1,
        ),
        log_dir=None,
    )

    train_sonic({"algo": {"max_iterations": 0}}, plan=plan, env=_StateEnv())

    assert applied == {"intra": 2, "interop": 1}


def test_device_resolution_honors_single_explicit_device() -> None:
    assert (
        _resolve_sonic_device(
            {"training": {"devices": [3]}},
            local_rank=0,
            world_size=1,
            cuda_available=True,
        )
        == "cuda:3"
    )
    assert (
        _resolve_sonic_device(
            {"training": {"devices": [3, 5]}},
            local_rank=1,
            world_size=2,
            cuda_available=True,
        )
        == "cuda:1"
    )


def test_device_resolution_rejects_invalid_distributed_topology() -> None:
    with pytest.raises(ValueError, match="local_rank"):
        _resolve_sonic_device(
            {"training": {"devices": [0, 1]}},
            local_rank=2,
            world_size=2,
            cuda_available=False,
        )
    with pytest.raises(ValueError, match="has 1 entries"):
        _resolve_sonic_device(
            {"training": {"devices": [0]}},
            local_rank=0,
            world_size=2,
            cuda_available=False,
        )


def _checkpoint_config() -> dict[str, object]:
    return {
        "actor_obs_dim": 1,
        "critic_obs_dim": 1,
        "tokenizer_obs_dim": 1,
        "action_dim": 1,
        "num_steps_per_env": 1,
        "num_mini_batches": 1,
        "num_learning_epochs": 1,
    }


def test_runner_loads_legacy_log_std_checkpoint(tmp_path) -> None:
    config = _checkpoint_config()
    runner = SonicPPORunner(_StateEnv(), config, device="cpu")
    state = {
        "model": {
            **runner.model.state_dict(),
            "log_std": torch.log(torch.full((1,), 0.2)),
        }
    }
    state["model"].pop("std")
    checkpoint = tmp_path / "legacy.pt"
    torch.save(state, checkpoint)
    runner.load(checkpoint)
    assert torch.allclose(runner.model.std, torch.full((1,), 0.2))


def test_runner_rejects_checkpoint_token_contract_mismatch(tmp_path) -> None:
    config = _checkpoint_config()
    runner = SonicPPORunner(_StateEnv(), config, device="cpu")
    checkpoint = tmp_path / "bad-token.pt"
    runner.save(checkpoint)
    state = torch.load(checkpoint, weights_only=False)
    state["token_info"]["num_tokens"] = 3
    state["contract"]["token_info"]["num_tokens"] = 3
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="token contract"):
        runner.load(checkpoint)


@pytest.mark.parametrize(
    "missing_field",
    (
        "actor_obs_dim",
        "critic_obs_dim",
        "tokenizer_obs_dim",
        "action_dim",
        "horizon",
        "token_info",
    ),
)
def test_runner_rejects_checkpoint_missing_required_contract_field(
    tmp_path, missing_field: str
) -> None:
    runner = SonicPPORunner(_StateEnv(), _checkpoint_config(), device="cpu")
    checkpoint = tmp_path / f"missing-{missing_field}.pt"
    runner.save(checkpoint)
    state = torch.load(checkpoint, weights_only=False)
    del state["contract"][missing_field]
    torch.save(state, checkpoint)

    with pytest.raises(ValueError, match="contract is missing fields"):
        runner.load(checkpoint)


def test_runner_rejects_non_mapping_checkpoint_contract(tmp_path) -> None:
    runner = SonicPPORunner(_StateEnv(), _checkpoint_config(), device="cpu")
    checkpoint = tmp_path / "non-mapping-contract.pt"
    runner.save(checkpoint)
    state = torch.load(checkpoint, weights_only=False)
    state["contract"] = None
    torch.save(state, checkpoint)

    with pytest.raises(ValueError, match="contract must be a mapping"):
        runner.load(checkpoint)


def test_runner_rejects_incomplete_checkpoint_token_info(tmp_path) -> None:
    runner = SonicPPORunner(_StateEnv(), _checkpoint_config(), device="cpu")
    checkpoint = tmp_path / "incomplete-token-info.pt"
    runner.save(checkpoint)
    state = torch.load(checkpoint, weights_only=False)
    del state["contract"]["token_info"]["total_dim"]
    torch.save(state, checkpoint)

    with pytest.raises(ValueError, match="token_info is missing fields"):
        runner.load(checkpoint)


def test_runner_treats_contractless_checkpoint_as_model_only(tmp_path) -> None:
    runner = SonicPPORunner(_StateEnv(), _checkpoint_config(), device="cpu")
    runner.current_learning_iteration = 4
    runner.algorithm.update_count = 3
    model_state = {key: value.clone() for key, value in runner.model.state_dict().items()}
    model_state["std"].fill_(0.2)
    checkpoint = tmp_path / "legacy-model-only.pt"
    torch.save(
        {
            "model": model_state,
            "optimizer": "must not be loaded",
            "algorithm": "must not be loaded",
            "iteration": 99,
            "token_info": {"num_tokens": 99},
        },
        checkpoint,
    )

    runner.load(checkpoint)

    assert torch.allclose(runner.model.std, torch.full((1,), 0.2))
    assert runner.current_learning_iteration == 4
    assert runner.algorithm.update_count == 3


@pytest.mark.parametrize("has_top_level_optimizer", (True, False))
def test_runner_loads_optimizer_once_with_top_level_priority(
    monkeypatch: pytest.MonkeyPatch, tmp_path, has_top_level_optimizer: bool
) -> None:
    runner = SonicPPORunner(_StateEnv(), _checkpoint_config(), device="cpu")
    checkpoint = tmp_path / "optimizer-owner.pt"
    runner.save(checkpoint)
    state = torch.load(checkpoint, weights_only=False)
    state["algorithm"]["optimizer"] = {"source": "algorithm"}
    if has_top_level_optimizer:
        state["optimizer"] = {"source": "top-level"}
    else:
        del state["optimizer"]
    torch.save(state, checkpoint)
    loaded: list[object] = []
    monkeypatch.setattr(runner.algorithm.optimizer, "load_state_dict", loaded.append)

    runner.load(checkpoint)

    expected_source = "top-level" if has_top_level_optimizer else "algorithm"
    assert loaded == [{"source": expected_source}]


class _TimeoutEnv:
    num_envs = 2

    @staticmethod
    def _observations(value: float) -> dict[str, np.ndarray]:
        return {
            "policy": np.full((2, 1), value, dtype=np.float32),
            "privileged": np.full((2, 1), value, dtype=np.float32),
            "tokens": np.full((2, 1), value, dtype=np.float32),
        }

    def init_state(self) -> NpEnvState:
        return NpEnvState(
            self._observations(0.0),
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=bool),
            np.zeros(2, dtype=bool),
            {},
        )

    def step(self, actions: np.ndarray) -> NpEnvState:
        assert actions.shape == (2, 1)
        return NpEnvState(
            self._observations(100.0),
            np.zeros(2, dtype=np.float32),
            np.array([False, True]),
            np.array([True, False]),
            {},
            final_observation=self._observations(3.0),
        )


def test_runner_bootstraps_timeout_from_final_observation(monkeypatch) -> None:
    config = {
        "actor_obs_dim": 1,
        "critic_obs_dim": 1,
        "tokenizer_obs_dim": 1,
        "action_dim": 1,
        "num_steps_per_env": 1,
        "num_mini_batches": 1,
        "num_learning_epochs": 1,
        "gamma": 0.9,
        "sonic": {
            "model": {
                "hidden_dims": [2],
                "tokenizer_hidden_dim": 2,
                "token_count": 1,
                "token_levels": 2,
            }
        },
    }
    runner = SonicPPORunner(_TimeoutEnv(), config, device="cpu")

    def distribution(actor_obs, critic_obs, token_obs):
        del actor_obs, token_obs
        policy = torch.distributions.Normal(torch.zeros((2, 1)), torch.ones((2, 1)))
        return policy, critic_obs[:, 0]

    captured: dict[str, torch.Tensor] = {}

    def update(storage) -> dict[str, float]:
        captured["rewards"] = storage.rewards.clone()
        captured["dones"] = storage.dones.clone()
        captured["returns"] = storage.returns.clone()
        storage.clear()
        return {"loss": 0.0}

    monkeypatch.setattr(runner.model, "distribution", distribution)
    monkeypatch.setattr(runner.algorithm, "update", update)

    metrics = runner.learn(1)

    assert metrics["loss"] == 0.0
    assert metrics["perf/rollout_fps"] > 0.0
    assert torch.equal(captured["dones"], torch.tensor([[True, True]]))
    assert torch.allclose(captured["rewards"], torch.tensor([[2.7, 0.0]]))
    assert torch.allclose(captured["returns"], torch.tensor([[2.7, 0.0]]))


def test_runner_rejects_wrong_observation_batch() -> None:
    config = {
        "actor_obs_dim": 1,
        "critic_obs_dim": 1,
        "tokenizer_obs_dim": 1,
        "action_dim": 1,
        "num_steps_per_env": 1,
        "num_mini_batches": 1,
        "num_learning_epochs": 1,
    }
    with pytest.raises(ValueError, match="batch shape"):
        SonicPPORunner(
            type("BadEnv", (), {"num_envs": 2, "init_state": lambda self: _obs(1)})(),
            config,
            device="cpu",
        )._reset()


def test_runner_preserves_gymnasium_final_observation_info() -> None:
    runner = SonicPPORunner(
        _StateEnv(),
        {"actor_obs_dim": 930, "critic_obs_dim": 1645, "tokenizer_obs_dim": 1761},
        device="cpu",
    )
    next_obs = _obs(2)
    final_obs = {
        "policy": np.ones((2, 930), dtype=np.float32),
        "privileged": np.ones((2, 1645), dtype=np.float32),
        "tokens": np.ones((2, 1761), dtype=np.float32),
    }
    parsed = runner._parse_step_result(
        (
            next_obs,
            np.zeros(2, dtype=np.float32),
            np.array([False, True]),
            np.array([True, False]),
            {"final_observation": final_obs},
        )
    )
    assert parsed[4] is final_obs
    assert torch.equal(parsed[2], torch.tensor([False, True]))
    assert torch.equal(parsed[3], torch.tensor([True, False]))
