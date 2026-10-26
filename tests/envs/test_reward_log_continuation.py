"""Cold reward-cache restoration never performs another environment step."""

import pytest
import torch

from unilab.managers.continuation import restore_tree

from .test_manager_based_rl_env import _make_cfg, _make_env


def test_first_step_reward_cache_restores_into_genuine_cold_environment():
    original, _ = _make_env(_make_cfg(include_optional_managers=False))
    restored, _ = _make_env(_make_cfg(include_optional_managers=False))
    try:
        original.init_state()
        original.step(torch.tensor([[0.25], [0.75]], dtype=torch.float32))
        restored.init_state()
        assert restored._reward_log_means.shape == (0,)
        values = {
            "_reward_log_names": original._reward_log_names,
            "_reward_log_means": original._reward_log_means.detach().clone(),
        }
        before_step = restored.step_counter
        before_rng = restored.rng.get_state().clone()
        restored._prepare_reward_log_continuation(values)
        for key, value in values.items():
            setattr(restored, key, restore_tree(getattr(restored, key), value, device="cpu"))
        assert restored.step_counter == before_step
        assert torch.equal(restored.rng.get_state(), before_rng)
        assert torch.equal(restored._reward_log_means, original._reward_log_means)
        assert restored.publish_reward_log()["reward/track"] == pytest.approx(0.5)
        storage = restored._reward_log_means.data_ptr()
        restored._prepare_reward_log_continuation(values)
        assert restored._reward_log_means.data_ptr() == storage
    finally:
        original.close()
        restored.close()


@pytest.mark.parametrize(
    "names,means",
    [
        (("unknown",), torch.zeros(1)),
        (("track",), torch.zeros(2)),
        (("track",), torch.zeros(1, dtype=torch.float64)),
        ((), torch.zeros(1)),
    ],
)
def test_reward_cache_rejects_unconfigured_layout(names, means):
    env, _ = _make_env(_make_cfg(include_optional_managers=False))
    try:
        with pytest.raises(ValueError, match="reward-log layout"):
            env._prepare_reward_log_continuation(
                {
                    "_reward_log_names": names,
                    "_reward_log_means": means,
                }
            )
    finally:
        env.close()
